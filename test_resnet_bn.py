import torch

import torch.distributed as dist
import pdb

from mmdet.models.backbones.resnet import ResNet
from torch.nn.modules.batchnorm import _BatchNorm

def show_bn(rn):
    for m in rn.modules():
        if isinstance(m, _BatchNorm):
            print(m.training)
            m.eval()
            print(m.training)
            return

rn1 = ResNet(depth=18, norm_eval=True)
rn2 = ResNet(depth=18, norm_eval=False)

pdb.set_trace()
