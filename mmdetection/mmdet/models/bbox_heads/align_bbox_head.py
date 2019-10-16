import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from mmdet.core import auto_fp16, force_fp32, mask_target
from ..builder import build_loss

from mmcv.cnn.weight_init import normal_init, xavier_init

from ..backbones.resnet import Bottleneck
from ..registry import HEADS
from ..utils import ConvModule
from .bbox_head import BBoxHead

# for mask
import mmcv
import numpy as np
import pycocotools.mask as mask_util


class AlignConv(nn.Module):
    def __init__(self, bbox_in_channel=256, mid_channel=256, bbox_out_channel=256, mask_in_channel=256, mask_out_channel=256, align_channel=1, with_gmp=True):
        super(AlignConv, self).__init__()
        self.change_channel = bbox_in_channel != bbox_out_channel
        self.relu = nn.ReLU(inplace=True)
        self.with_align = True
        self.align_channel = align_channel

        # bbox head
        if self.change_channel:
            self.bbox_res_1 = nn.Sequential(
                                nn.Conv2d(in_channels=bbox_in_channel, out_channels=bbox_out_channel, kernel_size=1, padding=0),
                                nn.BatchNorm2d(bbox_out_channel)
                            )
        else:
            self.bbox_res_1 = nn.Sequential(
                                nn.Conv2d(in_channels=bbox_in_channel, out_channels=mid_channel, kernel_size=1, padding=0),
                                nn.BatchNorm2d(mid_channel),
                                nn.ReLU(inplace=True)
                            )
        self.bbox_res_2 = nn.Sequential(
                                nn.Conv2d(in_channels=bbox_in_channel, out_channels=mid_channel, kernel_size=3, padding=1),
                                nn.BatchNorm2d(mid_channel),
                                nn.ReLU(inplace=True)
                            )
        self.bbox_res_3 = nn.Sequential(
                                nn.Conv2d(in_channels=mid_channel + self.with_align * self.align_channel, out_channels=bbox_out_channel, kernel_size=3, padding=1),
                                nn.BatchNorm2d(bbox_out_channel),
                            )

        # mask head
        self.mask_res_2 = nn.Sequential(
                                nn.Conv2d(in_channels=mask_in_channel, out_channels=mid_channel, kernel_size=3, padding=1),
                                nn.BatchNorm2d(mid_channel),
                                nn.ReLU(inplace=True)
                            )
        self.mask_res_3 = nn.Sequential(
                                nn.Conv2d(in_channels=mid_channel + self.with_align * self.align_channel, out_channels=mask_out_channel, kernel_size=1, padding=0),
                                nn.BatchNorm2d(mask_out_channel),
                                nn.ReLU(inplace=True)
                            )

        # align block
        if self.with_align:
            self.align_from_bbox = nn.Sequential(
                                nn.Conv2d(in_channels=bbox_in_channel, out_channels=self.align_channel, kernel_size=1, padding=0),
                                nn.BatchNorm2d(align_channel),
                                nn.ReLU(inplace=True)
                            )

            self.align_from_mask = nn.Sequential(
                                nn.Conv2d(in_channels=mask_in_channel, out_channels=self.align_channel, kernel_size=1, padding=0),
                                nn.BatchNorm2d(align_channel),
                                nn.ReLU(inplace=True)
                            )
    def forward(self, x_bbox, x_mask):
        if self.change_channel:
            identity = self.bbox_res_1(x_bbox)
        else:
            identity = x_bbox

        # align
        if self.with_align:
            alignment_from_bbox = self.align_from_bbox(x_bbox)
            alignment_from_bbox = (alignment_from_bbox.max(-1,keepdim=True)[0] + alignment_from_bbox.max(-2,keepdim=True)[0]) / 2

            alignment_from_mask = self.align_from_mask(x_mask)

        x_bbox = self.bbox_res_2(x_bbox)
        x_mask = self.mask_res_2(x_mask)
        if self.with_align:
            x_bbox = torch.cat([x_bbox, alignment_from_mask], dim=1)
            x_mask = torch.cat([x_mask, alignment_from_bbox], dim=1)

        x_bbox = self.relu(identity + self.bbox_res_3(x_bbox))
        x_mask = self.mask_res_3(x_mask)

        return x_bbox, x_mask

@HEADS.register_module
class AlignBBoxHead(BBoxHead):
    """Bbox head used in Double-Head R-CNN

                                      /-> cls
                  /-> shared convs ->
                                      \-> reg
    roi features
                                      /-> cls
                  \-> shared fc    ->
                                      \-> reg
    """  # noqa: W605

    def __init__(self,
                 num_convs=0,
                 num_fcs=0,
                 conv_out_channels=1024,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 # by wyz, for mask
                 #num_convs=4,#mask
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 #conv_out_channels=256,#mask
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 #conv_cfg=None,#mask
                 #norm_cfg=None,#mask
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)
        super(AlignBBoxHead, self).__init__(**kwargs)
        assert self.with_avg_pool
        assert num_convs > 0
        assert num_fcs > 0
        self.num_convs = num_convs
        self.num_fcs = num_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # 1.0 by wyz
        # increase the channel of input features
        #self.res_block = BasicResBlock(self.in_channels,self.conv_out_channels)

        # add conv heads
        self.conv_branch = self._add_conv_branch()
        # add fc heads
        self.fc_branch = self._add_fc_branch()

        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        #self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg)

        self.fc_reg = nn.Linear(self.conv_out_channels * 2 * 14, out_dim_reg)
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes)
        self.relu = nn.ReLU(inplace=True)



        # 2.0 by wyz, for mask
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        #self.conv_out_channels = conv_out_channels#mask
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)

        """
        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        """
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_in_channels = 256
            self.upsample = nn.ConvTranspose2d(
                upsample_in_channels,
                256, #self.conv_out_channels,
                self.upsample_ratio,
                stride=self.upsample_ratio)
        else:
            self.upsample = nn.Upsample(
                scale_factor=self.upsample_ratio, mode=self.upsample_method)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            256 #self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits_mask = nn.Conv2d(logits_in_channel, out_channels, 1)

    def _add_conv_branch(self):
        """Add the fc branch which consists of a sequential of conv layers"""
        branch_convs = nn.ModuleList()
        last_channel = self.in_channels
        for i in range(self.num_convs):
            branch_convs.append(
                AlignConv(bbox_in_channel=last_channel, mid_channel=256, bbox_out_channel=self.conv_out_channels, mask_in_channel=256, mask_out_channel=256, align_channel=1, with_gmp=True)
            )
            """
                Bottleneck(
                    inplanes=self.conv_out_channels,
                    planes=self.conv_out_channels // 4,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            """
            last_channel = self.conv_out_channels
        return branch_convs

    def _add_fc_branch(self):
        """Add the fc branch which consists of a sequential of fc layers"""
        branch_fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                self.in_channels *
                self.roi_feat_area if i == 0 else self.fc_out_channels)
            branch_fcs.append(nn.Linear(fc_in_channels, self.fc_out_channels))
        return branch_fcs

    def init_weights(self):
        normal_init(self.fc_cls, std=0.01)
        normal_init(self.fc_reg, std=0.001)

        for m in self.fc_branch.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')


        for m in [self.upsample, self.conv_logits_mask]:
            if m is None:
                continue
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

    # by wyz, from mask
    def get_target_mask(self, sampling_results, gt_masks, rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        pos_assigned_gt_inds = [
            res.pos_assigned_gt_inds for res in sampling_results
        ]
        mask_targets = mask_target(pos_proposals, pos_assigned_gt_inds,
                                   gt_masks, rcnn_train_cfg)
        return mask_targets

    def forward(self, x, last_feat=None, pos_inds=None):# x_cls, x_reg): # from double head
        x_clss = x
        x_bbox = x
        x_mask = x
        if pos_inds is not None:
            x_bbox = x_bbox[pos_inds]
            x_mask = x_mask[pos_inds]

        x_clss = F.interpolate(x_clss, size=(7,7), mode="bilinear", align_corners=None)
        x_clss = x_clss.view(x_clss.shape[0], -1)
        for fc in self.fc_branch:
            x_clss = self.relu(fc(x_clss))
        for conv in self.conv_branch:
            x_bbox, x_mask = conv(x_bbox, x_mask)

        # clss
        cls_score = self.fc_cls(x_clss)
        # bbox
        x_bbox = torch.cat([x_bbox.max(-1)[0], x_bbox.max(-2)[0]], dim=-1).view(x_bbox.shape[0], -1)
        bbox_pred = self.fc_reg(x_bbox)
        # mask
        x_mask = self.upsample(x_mask)
        mask_pred = self.conv_logits_mask(x_mask)

        return cls_score, bbox_pred, mask_pred

        """
        # in double head
        # conv head
        x_conv = self.res_block(x_reg)

        for conv in self.conv_branch:
            x_conv = conv(x_conv)

        if self.with_avg_pool:
            x_conv = self.avg_pool(x_conv)

        x_conv = x_conv.view(x_conv.size(0), -1)
        bbox_pred = self.fc_reg(x_conv)

        # fc head
        x_fc = x_cls.view(x_cls.size(0), -1)
        for fc in self.fc_branch:
            x_fc = self.relu(fc(x_fc))

        cls_score = self.fc_cls(x_fc)

        return cls_score, bbox_pred
        """

    @force_fp32(apply_to=('mask_pred', ))
    def loss_for_mask(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask
        return loss

    def get_seg_masks(self, mask_pred, det_bboxes, det_labels, rcnn_test_cfg,
                      ori_shape, scale_factor, rescale):
        """Get segmentation masks from mask_pred and bboxes.

        Args:
            mask_pred (Tensor or ndarray): shape (n, #class+1, h, w).
                For single-scale testing, mask_pred is the direct output of
                model, whose type is Tensor, while for multi-scale testing,
                it will be converted to numpy array outside of this method.
            det_bboxes (Tensor): shape (n, 4/5)
            det_labels (Tensor): shape (n, )
            img_shape (Tensor): shape (3, )
            rcnn_test_cfg (dict): rcnn testing config
            ori_shape: original image size

        Returns:
            list[list]: encoded masks
        """
        if isinstance(mask_pred, torch.Tensor):
            mask_pred = mask_pred.sigmoid().cpu().numpy()
        assert isinstance(mask_pred, np.ndarray)
        # when enabling mixed precision training, mask_pred may be float16
        # numpy array
        mask_pred = mask_pred.astype(np.float32)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        bboxes = det_bboxes.cpu().numpy()[:, :4]
        labels = det_labels.cpu().numpy() + 1

        if rescale:
            img_h, img_w = ori_shape[:2]
        else:
            img_h = np.round(ori_shape[0] * scale_factor).astype(np.int32)
            img_w = np.round(ori_shape[1] * scale_factor).astype(np.int32)
            scale_factor = 1.0

        for i in range(bboxes.shape[0]):
            bbox = (bboxes[i, :] / scale_factor).astype(np.int32)
            label = labels[i]
            w = max(bbox[2] - bbox[0] + 1, 1)
            h = max(bbox[3] - bbox[1] + 1, 1)

            if not self.class_agnostic:
                mask_pred_ = mask_pred[i, label, :, :]
            else:
                mask_pred_ = mask_pred[i, 0, :, :]
            im_mask = np.zeros((img_h, img_w), dtype=np.uint8)

            bbox_mask = mmcv.imresize(mask_pred_, (w, h))
            bbox_mask = (bbox_mask > rcnn_test_cfg.mask_thr_binary).astype(
                np.uint8)
            im_mask[bbox[1]:bbox[1] + h, bbox[0]:bbox[0] + w] = bbox_mask
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label - 1].append(rle)

        return cls_segms

