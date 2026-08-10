"""
PyTorch BiSeNetV1 implementation matching the production ONNX model.

Architecture: BiSeNetV1 with ResNet-18 backbone, 19 CelebAMask-HQ classes.
Source: https://github.com/zllrunning/face-parsing.PyTorch

This is a faithful reproduction of the upstream model definition,
verified against the production ONNX graph structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm2d + ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class BasicBlock(nn.Module):
    """ResNet BasicBlock: conv3x3 → BN → ReLU → conv3x3 → BN → (+shortcut) → ReLU."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


def _make_layer(
    block: type[BasicBlock],
    in_channels: int,
    out_channels: int,
    blocks: int,
    stride: int,
) -> nn.Sequential:
    """Build a ResNet layer with optional downsampling."""
    downsample = None
    if stride != 1 or in_channels != out_channels * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * block.expansion),
        )

    layers: list[nn.Module] = []
    layers.append(block(in_channels, out_channels, stride, downsample))
    for _ in range(1, blocks):
        layers.append(block(out_channels, out_channels))

    return nn.Sequential(*layers)


class ResNet18(nn.Module):
    """ResNet-18 backbone returning features at 1/8, 1/16, 1/32 scales."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = _make_layer(BasicBlock, 64, 64, 2, stride=1)     # 64 ch, 1/4
        self.layer2 = _make_layer(BasicBlock, 64, 128, 2, stride=2)    # 128 ch, 1/8
        self.layer3 = _make_layer(BasicBlock, 128, 256, 2, stride=2)   # 256 ch, 1/16
        self.layer4 = _make_layer(BasicBlock, 256, 512, 2, stride=2)   # 512 ch, 1/32

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x)     # 1/8 scale, 128 channels
        feat16 = self.layer3(feat8) # 1/16 scale, 256 channels
        feat32 = self.layer4(feat16) # 1/32 scale, 512 channels

        return feat8, feat16, feat32


class AttentionRefinementModule(nn.Module):
    """ARM: channel attention via global average pooling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_block = ConvBNReLU(in_channels, out_channels, kernel_size=3)
        self.conv_atten = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.conv_block(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])  # global avg pool → (B,C,1,1)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid(atten)
        return torch.mul(feat, atten)


class ContextPath(nn.Module):
    """Context path with ARM32, ARM16, and cascaded upsampling."""

    def __init__(self) -> None:
        super().__init__()
        self.resnet = ResNet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, kernel_size=3)
        self.conv_head16 = ConvBNReLU(128, 128, kernel_size=3)
        self.conv_avg = ConvBNReLU(512, 128, kernel_size=1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W = x.size()[2], x.size()[3]
        feat_res8, feat_res16, feat_res32 = self.resnet(x)

        # Global context
        avg = self.conv_avg(F.avg_pool2d(feat_res32, feat_res32.size()[2:]))

        # ARM32 → add global context → upsample
        feat32_arm = self.arm32(feat_res32)
        feat32_sum = feat32_arm + avg
        feat32_up = self.conv_head32(
            F.interpolate(feat32_sum, size=(H // 16, W // 16), mode="nearest")
        )

        # ARM16 → add upsampled context → upsample
        feat16_arm = self.arm16(feat_res16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.conv_head16(
            F.interpolate(feat16_sum, size=(H // 8, W // 8), mode="nearest")
        )

        return feat_res8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    """FFM: fuses spatial and context features with channel attention gating."""

    def __init__(self, in_channels: int = 256, out_channels: int = 256) -> None:
        super().__init__()
        self.convblk = ConvBNReLU(in_channels, out_channels, kernel_size=1)
        self.conv1 = nn.Conv2d(out_channels, out_channels // 4, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels // 4, out_channels, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(
        self, fsp: torch.Tensor, fcp: torch.Tensor
    ) -> torch.Tensor:
        fcat = torch.cat([fsp, fcp], dim=1)        # 128+128=256 channels
        feat = self.convblk(fcat)                    # 1×1 Conv+BN+ReLU → 256ch
        atten = F.avg_pool2d(feat, feat.size()[2:])  # global avg pool
        atten = self.relu(self.conv1(atten))          # 256→64
        atten = self.sigmoid(self.conv2(atten))       # 64→256
        return torch.mul(feat, atten) + feat          # gated + residual


class BiSeNetOutput(nn.Module):
    """Output head: ConvBNReLU → Conv2d → upsample to input size."""

    def __init__(self, in_channels: int, mid_channels: int, n_classes: int) -> None:
        super().__init__()
        self.convblk = ConvBNReLU(in_channels, mid_channels, kernel_size=3)
        self.conv = nn.Conv2d(mid_channels, n_classes, 1, bias=False)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.convblk(x)
        x = self.conv(x)
        x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)
        return x


class BiSeNet(nn.Module):
    """BiSeNetV1 for face parsing.

    Produces 3 outputs: main, auxiliary@1/16, auxiliary@1/32.
    During inference, only the main output is used.
    """

    def __init__(self, n_classes: int = 19) -> None:
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W = x.size()[2], x.size()[3]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_fuse = self.ffm(feat_res8, feat_cp8)

        out = self.conv_out(feat_fuse, H, W)
        out16 = self.conv_out16(feat_cp8, H, W)
        out32 = self.conv_out32(feat_cp16, H, W)

        return out, out16, out32
