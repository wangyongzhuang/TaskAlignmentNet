import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from mmdet.core import auto_fp16, force_fp32, mask_target
from ..builder import build_loss

from mmcv.cnn.weight_init import normal_init, xavier_init

from ..backbones.resnet import Bottleneck
from ..registry import HEADS
from ..utils import ConvModule, build_conv_layer, build_norm_layer
from .bbox_head import BBoxHead
from ..losses import accuracy

# for mask
import mmcv
import numpy as np
import pycocotools.mask as mask_util

class BasicResBlock(nn.Module):
    """Basic residual block.

    This block is a little different from the block in the ResNet backbone.
    The kernel size of conv1 is 1 in this block while 3 in ResNet BasicBlock.

    Args:
        in_channels (int): Channels of the input feature map.
        out_channels (int): Channels of the output feature map.
        conv_cfg (dict): The config dict for convolution layers.
        norm_cfg (dict): The config dict for normalization layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN')):
        super(BasicResBlock, self).__init__()

        # main path
        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            bias=False,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
        self.conv2 = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            activation=None,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)

        # identity path
        self.conv_identity = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            activation=None)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.conv2(x)

        identity = self.conv_identity(identity)
        out = x + identity

        out = self.relu(out)
        return out



class Bottleneck_example(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes=1024,
                 planes=256,
                 mask_inplanes=256,
                 stride=1,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 ):

        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck_example, self).__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_cfg_align = norm_cfg
        if "GN" == self.norm_cfg_align["type"].upper():
            self.norm_cfg_align["num_groups"] = 1

        self.conv1_stride = 1
        self.conv2_stride = stride

        self.with_align = True #True
        self.with_align_from_bbox_to_mask = True
        self.with_align_from_mask_to_bbox = True
        self.align_channel = 1 #1
        print("with_align: {} box2mask: {}, mask2box: {}, channel: {}, norm:".format(self.with_align, self.with_align_from_bbox_to_mask, self.with_align_from_mask_to_bbox, self.align_channel), self.norm_cfg)

        # bbox
        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, planes, postfix=1)#
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, planes, postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes + self.with_align * self.with_align_from_mask_to_bbox * self.align_channel,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.norm3_name, norm3 = build_norm_layer(self.norm_cfg, planes * self.expansion, postfix=3)
        self.add_module(self.norm3_name, norm3)

        # mask
        self.conv2_mask = build_conv_layer(
                conv_cfg,
                mask_inplanes,
                mask_inplanes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        self.norm4_name, norm4 = build_norm_layer(self.norm_cfg, mask_inplanes, postfix=4)
        self.add_module(self.norm4_name, norm4)

        if self.with_align:
            self.conv3_mask = build_conv_layer(
                conv_cfg,
                mask_inplanes + self.with_align * self.with_align_from_bbox_to_mask * self.align_channel,
                mask_inplanes,
                kernel_size=1,
                bias=False)
            self.norm5_name, norm5 = build_norm_layer(self.norm_cfg, mask_inplanes, postfix=5)
            self.add_module(self.norm5_name, norm5)

            if self.with_align_from_bbox_to_mask:
                self.align_from_bbox = build_conv_layer(
                    conv_cfg,
                    planes,
                    self.align_channel,
                    kernel_size=1,
                    bias=False)
                self.norm6_name, norm6 = build_norm_layer(self.norm_cfg_align, self.align_channel, postfix=6)
                self.add_module(self.norm6_name, norm6)

            if self.with_align_from_mask_to_bbox:
                self.align_from_mask = build_conv_layer(
                    conv_cfg,
                    mask_inplanes,
                    self.align_channel,
                    kernel_size=1,
                    bias=False)
                self.norm7_name, norm7 = build_norm_layer(self.norm_cfg_align, self.align_channel, postfix=7)
                self.add_module(self.norm7_name, norm7)

        self.relu = nn.ReLU(inplace=True)



    def forward(self, x_bbox, x_mask):

        def _inner_forward(x_bbox, x_mask):
            identity_bbox = x_bbox

            out = self.conv1(x_bbox)
            out = getattr(self, self.norm1_name)(out)
            out = self.relu(out)

            if self.with_align:
                if self.with_align_from_bbox_to_mask:
                    alignment_from_bbox = self.align_from_bbox(out)
                    alignment_from_bbox = getattr(self, self.norm6_name)(alignment_from_bbox)
                    alignment_from_bbox = self.relu(alignment_from_bbox)
                    alignment_from_bbox = (alignment_from_bbox.max(-1,keepdim=True)[0] + alignment_from_bbox.max(-2,keepdim=True)[0]) / 2

                if self.with_align_from_mask_to_bbox:
                    alignment_from_mask = self.align_from_mask(x_mask)
                    alignment_from_mask = getattr(self, self.norm7_name)(alignment_from_mask)
                    alignment_from_mask = self.relu(alignment_from_mask)

            out = self.conv2(out)
            out = getattr(self, self.norm2_name)(out)
            out = self.relu(out)


            out_mask = self.conv2_mask(x_mask)
            out_mask = getattr(self, self.norm4_name)(out_mask)
            out_mask = self.relu(out_mask)

            if self.with_align:
                if self.with_align_from_mask_to_bbox:
                    out = torch.cat([out, alignment_from_mask], dim=1)
                if self.with_align_from_bbox_to_mask:
                    out_mask = torch.cat([out_mask, alignment_from_bbox], dim=1)

                out_mask = self.conv3_mask(out_mask)
                out_mask = getattr(self, self.norm5_name)(out_mask)
                out_mask = self.relu(out_mask)


            out = self.conv3(out)
            out = getattr(self, self.norm3_name)(out)
            out += identity_bbox
            out = self.relu(out)

            return out, out_mask

        out_bbox, out_mask = _inner_forward(x_bbox, x_mask)


        return out_bbox, out_mask


@HEADS.register_module
class AlignBBoxHead(BBoxHead):
    def __init__(self,
                 num_convs=4,
                 num_fcs=2,
                 conv_out_channels=1024,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 # by wyz, for mask
                 roi_feat_size=14,
                 in_channels=256,
                 conv_kernel_size=3,
                 #conv_out_channels=256,#mask
                 upsample_method='deconv',
                 upsample_ratio=2,
                 num_classes=81,
                 class_agnostic=False,
                 loss_mask=dict(
                     type='CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 **kwargs):
        kwargs.setdefault('with_avg_pool', True)#True
        super(AlignBBoxHead, self).__init__(**kwargs)
        #assert self.with_avg_pool
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
        #self.norm_cfg["affine"] = False

        self.with_misalign_loss = True #True
        print("norm cfg: ", self.norm_cfg)
        print("with misalign loss: ", self.with_misalign_loss)

        # 1.0 roi_feat
        # increase the channel of input features
        self.res_block = BasicResBlock(self.in_channels, self.conv_out_channels, norm_cfg=self.norm_cfg)

        self.roi_feat_size = _pair(roi_feat_size)
        # add conv heads
        self.align_branch = self._add_align_branch()
        #self.conv_branch = self._add_conv_branch()
        # add fc heads
        self.fc_branch = self._add_fc_branch()


        # 1.1 cls
        self.fc_cls = nn.Linear(self.fc_out_channels, self.num_classes)

        # 1.2 reg
        out_dim_reg = 4 if self.reg_class_agnostic else 4 * self.num_classes
        self.relu = nn.ReLU(inplace=True)
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
            print("avg_pool in the branch:{}".format(self.with_avg_pool))
            self.fc_reg = nn.Linear(self.conv_out_channels, out_dim_reg) # TODO # 14
        else:
            self.fc_reg = nn.Linear(self.conv_out_channels * 2 * 7, out_dim_reg) # TODO # 14

        # 1.3 mask
        if upsample_method not in [None, 'deconv', 'nearest', 'bilinear']:
            raise ValueError(
                'Invalid upsample method {}, accepted methods '
                'are "deconv", "nearest", "bilinear"'.format(upsample_method))
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        #self.conv_out_channels = conv_out_channels#mask
        self.upsample_method = upsample_method
        self.upsample_ratio = upsample_ratio
        self.fp16_enabled = False
        self.loss_mask = build_loss(loss_mask)

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

    def _add_align_branch(self):
        branch_convs = nn.ModuleList()
        #last_channel = self.in_channels
        last_channel = self.conv_out_channels
        for i in range(self.num_convs):
            branch_convs.append(
                Bottleneck_example(norm_cfg=self.norm_cfg)
            )
            last_channel = self.conv_out_channels
        return branch_convs

    def _add_fc_branch(self):
        """Add the fc branch which consists of a sequential of fc layers"""
        branch_fcs = nn.ModuleList()
        for i in range(self.num_fcs):
            fc_in_channels = (
                    self.in_channels *
                    49 if i == 0 else self.fc_out_channels) # TODO
                    #self.roi_feat_area if i == 0 else self.fc_out_channels)
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

    def forward(self, x, bbox_feats_7=None, last_feat=None, pos_inds=None):# x_cls, x_reg): # from double head
        if bbox_feats_7 is None:
            x_clss = x
        else:
            x_clss = bbox_feats_7
        x_bbox = x
        x_mask = x
        if pos_inds is not None:
            x_bbox = x_bbox[pos_inds]
            x_mask = x_mask[pos_inds]

        if bbox_feats_7 is None:
            x_clss = F.interpolate(x_clss, size=(7,7), mode="bilinear", align_corners=False)
        x_clss = x_clss.view(x_clss.shape[0], -1)
        for fc in self.fc_branch:
            x_clss = self.relu(fc(x_clss))

        #x_bbox_conv = F.interpolate(x_bbox, size=(7,7), mode="bilinear", align_corners=False)
        #x_bbox_conv = self.res_block(x_bbox_conv)
        #for conv in self.conv_branch:
        #    x_bbox_conv = conv(x_bbox_conv)
        #x_bbox = F.interpolate(x_bbox, size=(7,7), mode="bilinear", align_corners=False)
        
        x_bbox = self.res_block(x_bbox)
        for conv in self.align_branch:
            x_bbox, x_mask = conv(x_bbox, x_mask)
        #x_bbox = x_bbox_conv

        # clss
        cls_score = self.fc_cls(x_clss)
        # bbox
        #x_bbox = torch.cat([x_bbox.max(-1)[0], x_bbox.max(-2)[0]], dim=-1).view(x_bbox.shape[0], -1)
        # Note: AVG is better than GMP+AVG
        # TODO: only GMP
        #x_bbox = (x_bbox.max(-1, keepdim=True)[0] + x_bbox.max(-2, keepdim=True)[0]) / 2

        if self.with_avg_pool:
            x_bbox = self.avg_pool(x_bbox)
            x_bbox = x_bbox.view(x_bbox.size(0), -1)
        bbox_pred = self.fc_reg(x_bbox)
        # mask
        if self.upsample is not None:
            x_mask = self.upsample(x_mask)
            if self.upsample_method == 'deconv':
                x_mask = self.relu(x_mask)
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

    def transform_to_directional_mask(self, mask, bin_num=4):
        resolution = mask.size()[-1]
        mask_directional = []
        bin_size = resolution / bin_num
        for i in range(bin_num):
            for j in range(bin_num):
                idx_begin_i, idx_end_i = round(i*bin_size), round((i+1)*bin_size)
                idx_begin_j, idx_end_j = round(j*bin_size), round((j+1)*bin_size)
                mask_slice = mask[:,idx_begin_i:idx_end_i, idx_begin_j:idx_end_j]
                mask_directional.append(mask_slice.max(-1)[0])
                mask_directional.append(mask_slice.max(-2)[0])
        mask_directional = torch.cat(mask_directional, dim=-1)
        return mask_directional

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
    def loss(self,
             cls_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             reduction_override=None,
             pred_with_pos_inds=False):
        losses = dict()
        if cls_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                cls_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(cls_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            #import pdb
            #pdb.set_trace()
            if pred_with_pos_inds:
                pos_inds_for_reg = pos_inds[pos_inds]
            else:
                pos_inds_for_reg = pos_inds
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds_for_reg]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds_for_reg, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override) #* (1 + 0.3 * self.with_misalign_loss)
        return losses

    @force_fp32(apply_to=('mask_pred', ))
    def loss_for_mask(self, mask_pred, mask_targets, labels):
        loss = dict()
        if self.class_agnostic:
            loss_mask = self.loss_mask(mask_pred, mask_targets,
                                       torch.zeros_like(labels))
        else:
            loss_mask = self.loss_mask(mask_pred, mask_targets, labels)
        loss['loss_mask'] = loss_mask

        if self.with_misalign_loss:
            num_rois = mask_pred.size()[0]
            inds = torch.arange(0, num_rois, dtype=torch.long, device=mask_pred.device)
            pred_slice = mask_pred[inds, labels].squeeze(1)
            pred_slice_directional = self.transform_to_directional_mask(pred_slice, bin_num=4)
            target_directional = self.transform_to_directional_mask(mask_targets, bin_num=4)
            loss_align = F.binary_cross_entropy_with_logits(pred_slice_directional, target_directional, reduction='mean')[None]
            loss['loss_align'] = loss_align
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

