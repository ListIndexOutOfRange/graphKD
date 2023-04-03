""" Resnet in PyTorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

from __future__ import annotations
from torch import Tensor, nn
from functools import partial

Conv1x1 = partial(nn.Conv2d, kernel_size=1, bias=False)
Conv3x3 = partial(nn.Conv2d, kernel_size=3, bias=False)


class BasicBlock(nn.Module):

    """Basic Block for resnet 18 and resnet 34. """

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            Conv3x3(in_channels, out_channels, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv3x3(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                Conv1x1(in_channels, out_channels * BasicBlock.expansion, stride=stride),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x: Tensor) -> Tensor:
        return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))


class BottleNeck(nn.Module):

    """ Residual block for resnet over 50 layers. """

    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            Conv1x1(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv3x3(out_channels, out_channels, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv1x1(out_channels, out_channels * BottleNeck.expansion),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                Conv1x1(in_channels, out_channels * BottleNeck.expansion, stride=stride),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        return nn.ReLU(inplace=True)(self.residual(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block: nn.Module, num_blocks: list[int], num_classes: int = 100) -> None:
        super().__init__()
        self.in_channels = 64
        self.stem = nn.Sequential(
            Conv3x3(3, 64, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layers = nn.ModuleList()
        widths = [64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        for w, s, n in zip(widths, strides, num_blocks):
            self.layers.append(self._make_layer(block, w, n, s))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512 * block.expansion, num_classes)
        )

    def _make_layer(
        self, block: nn.Module, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: Tensor, return_all_features: bool = False) -> Tensor:
        f = self.stem(x)
        all_features = [f]
        for layer in self.layers:
            f = layer(f)
            all_features.append(f)
        f = self.head(f)
        all_features.append(f)
        return all_features if return_all_features else all_features[-1]

    @classmethod
    def from_name(cls, name: str, num_classes: int = 100) -> ResNet:
        config = dict(
            resnet18=dict(block=BasicBlock, num_blocks=[2, 2, 2, 2]),
            resnet34=dict(block=BasicBlock, num_blocks=[3, 4, 6, 3]),
            resnet50=dict(block=BottleNeck, num_blocks=[3, 4, 6, 3]),
            resnet101=dict(block=BottleNeck, num_blocks=[3, 4, 23, 3]),
            resnet152=dict(block=BottleNeck, num_blocks=[3, 8, 36, 3]),

        )
        return cls(**dict(**config[name], num_classes=num_classes))
