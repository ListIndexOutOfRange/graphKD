from __future__ import annotations
from torch import nn, Tensor


# _______________________________________________________________________________________________ #

class ConvNormAct(nn.Module):

    def __init__(
        self, in_features: int, out_features: int,
        kernel_size: int = 3, stride: int = 1, padding: int = 1, groups: int = 1, act: bool = False
    ) -> None:
        super(ConvNormAct, self).__init__()
        conv_params = dict(stride=stride, padding=padding, groups=groups, bias=False)
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, **conv_params)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        y = self.norm(self.conv(x))
        return y.relu() if self.act else y


class BasicBlock(nn.Module):

    def __init__(self, in_features: int, out_features: int, stride: int = 1) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = ConvNormAct(in_features, out_features, stride=stride, act=True)
        self.conv2 = ConvNormAct(out_features, out_features)
        if stride == 1 and in_features == out_features:
            self.shortcut = nn.Identity()
        else:
            conv_params = dict(kernel_size=1, stride=stride, padding=0)
            self.shortcut = ConvNormAct(in_features, out_features, **conv_params)

    def forward(self, x: Tensor) -> Tensor:
        z = self.conv2(self.conv1(x))
        z += self.shortcut(x)
        return z.relu()


# _______________________________________________________________________________________________ #

class ResNet(nn.Module):

    def __init__(
        self, block_params_list: list[tuple[int]], first_feature_maps, large: bool = False
    ) -> None:
        """
        Args:
            block_params_list (list[tuple[int]]): depth, stride, and expansion of each block.
            first_feature_maps (_type_):
                initial number of feature maps in first embedding, used as a base downstream
            large (bool, optional):
                - if True, embedding is a conv with kernel 7 followed by a maxpool
                - if False, embedding is a conv with kernel 3
                Defaults to False.
        """
        super(ResNet, self).__init__()
        if not large:
            first_conv_params = dict(kernel_size=3, stride=1, padding=1, act=True)
            self.pool = nn.Identity()
        else:
            first_conv_params = dict(kernel_size=7, stride=2, padding=3, act=True)
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.embedding = ConvNormAct(3, first_feature_maps, **first_conv_params)
        self.large = large
        blocks = list()
        last_multiplier = 1
        for (depth, stride, multiplier) in block_params_list:
            for i in range(depth):
                in_features = int(first_feature_maps * last_multiplier)
                out_features = int(first_feature_maps * multiplier)
                stride = 1 if i > 0 else stride
                blocks.append(BasicBlock(in_features, out_features, stride))
                last_multiplier = multiplier
        self.blocks = nn.ModuleList(blocks)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor, return_all_features: bool = False) -> Tensor:
        z = self.pool(self.embedding(x))
        all_features = [z]
        for block in self.blocks:
            z = block(z)
            all_features.append(z)
        return all_features[-1] if not return_all_features else all_features


# _______________________________________________________________________________________________ #


CONFIGS = dict(
    params=dict(
        resnet18=[(2, 1, 1), (2, 2, 2), (2, 2, 4), (2, 2, 8)],
        resnet20=[(3, 1, 1), (3, 2, 2), (3, 2, 4)],
        resnet56=[(9, 1, 1), (9, 2, 2), (9, 2, 4)],
        resnet56flat=[(9, 1, 1), (9, 2, 1.41), (9, 2, 2)],
        resnet110=[(18, 1, 1), (18, 2, 2), (18, 2, 4)],
        wrn28_10=[(4, 1, 10), (4, 2, 20), (4, 2, 40)],
        wrn16_16=[(2, 1, 16), (2, 2, 32), (2, 2, 64)],
    ),
    output_coeff=dict(
        resnet18=8,
        resnet20=4,
        resnet56=4,
        resnet56flat=2,
        resnet110=4,
        wrn28_10=40,
        wrn16_16=64,
    )
)


def make_resnet(name: str, first_feature_maps: int, large: bool) -> tuple[ResNet, int]:
    """ returns model, latent_dim. """
    latent_dim = CONFIGS["output_coeff"][name] * first_feature_maps
    return ResNet(CONFIGS["params"][name], first_feature_maps, large), latent_dim


def make_classifier(latent_dim: int, num_classes: int) -> int:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(latent_dim, num_classes)
    )
