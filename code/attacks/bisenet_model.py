"""
bisenet_model.py
================
BiSeNet architecture for face parsing.
Compatible with pretrained weights from face-parsing.PyTorch
(https://github.com/zllrunning/face-parsing.PyTorch).

Checkpoint: 79999_iter.pth  (19-class face parsing)

Labels:
    0: background    1: skin        2: l_brow      3: r_brow
    4: l_eye         5: r_eye       6: eye_g       7: l_ear
    8: r_ear         9: ear_r      10: nose       11: mouth
   12: u_lip        13: l_lip      14: neck       15: neck_l
   16: cloth        17: hair       18: hat
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv_out(self.conv(x))


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.sigmoid_atten(self.bn_atten(self.conv_atten(atten)))
        return feat * atten


class Resnet18(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        resnet = models.resnet18(weights=None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        feat4 = self.layer1(x)
        feat8 = self.layer2(feat4)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)

        avg = self.conv_avg(F.adaptive_avg_pool2d(feat32, 1))
        avg_up = F.interpolate(avg, feat32.shape[2:], mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = self.conv_head32(
            F.interpolate(feat32_sum, feat16.shape[2:], mode='nearest'))

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.conv_head16(
            F.interpolate(feat16_sum, feat8.shape[2:], mode='nearest'))

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        feat = self.convblk(torch.cat([fsp, fcp], dim=1))
        atten = F.adaptive_avg_pool2d(feat, 1)
        atten = self.sigmoid(self.conv2(self.relu(self.conv1(atten))))
        return feat * atten + feat


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19, *args, **kwargs):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.shape[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        out = F.interpolate(self.conv_out(feat_fuse), (H, W),
                            mode='bilinear', align_corners=True)
        out16 = F.interpolate(self.conv_out16(feat_cp8), (H, W),
                              mode='bilinear', align_corners=True)
        out32 = F.interpolate(self.conv_out32(feat_cp16), (H, W),
                              mode='bilinear', align_corners=True)
        return out, out16, out32
