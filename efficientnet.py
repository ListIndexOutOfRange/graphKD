""" A minimalist Pytorch implementation of EfficientNet directly based on the original paper:
    "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
    Mingxing Tan, Quoc V. Le, 2019, https://arxiv.org/abs/1905.11946
"""

from __future__ import annotations
import math
import torch
from torch import nn, Tensor


# +------------------------------------------------------------------------------------------+ #
# |                                      BUILDING BLOCKS                                     | #
# +------------------------------------------------------------------------------------------+ #

class ConvBnAct(nn.Sequential):

    """ Layer grouping a convolution, a batchnorm, and optionaly an activation function.

    Quoting original paper Section 5.2: "We train our EfficientNet models with [...] batch norm
    momentum 0.99 [...]. We also use SiLU (Swish-1) activation (Ramachandran et al., 2018;
    Elfwing et al., 2018; Hendrycks & Gimpel, 2016).
    """

    def __init__(
        self, n_in: int, n_out: int, kernel_size: int = 3, stride: int = 1, padding: int = 0,
        groups: int = 1, bias: bool = False, bn: bool = True, act: bool = True
    ) -> None:
        super().__init__()
        conv_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding,
                       'groups': groups, 'bias': bias}
        self.add_module('conv', nn.Conv2d(n_in, n_out, **conv_params))
        self.add_module('bn', nn.BatchNorm2d(n_out, momentum=0.99) if bn else nn.Identity())
        self.add_module('act', nn.SiLU() if act else nn.Identity())


class SEBlock(nn.Module):

    """ Squeeze-and-excitation block. """

    def __init__(self, n_in: int, r: int = 24) -> None:
        super().__init__()
        self.squeeze    = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(n_in, n_in // r, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(n_in // r, n_in, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.excitation(self.squeeze(x))


# class DropSample2d(nn.Module):

#     """ Drops each sample in x with probability p during training (a sort of DropConnect).

#     In the original paper Dropout regularization is mentionned but it's not what the official
#     repo shows. See this discussion: https://github.com/tensorflow/tpu/issues/494.

#     Official tensorflow code:
#     https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/utils.py#L276
#     """

#     def __init__(self, p: float = 0.) -> None:
#         super().__init__()
#         self.p = p

#     def forward(self, x: Tensor) -> Tensor:
#         if (not self.p) or (not self.training):
#             return x
#         random_tensor = torch.FloatTensor(len(x), 1, 1, 1).uniform_().to(x.device)
#         bit_mask = self.p < random_tensor
#         x = x.div(1 - self.p)
#         x = x * bit_mask
#         return x


def drop_connect(inputs: Tensor, p: float, training: bool) -> Tensor:
    """ Drop connect.

    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


# +------------------------------------------------------------------------------------------+ #
# |                                        MBCONV BLOCK                                      | #
# +------------------------------------------------------------------------------------------+ #

class MBConv(nn.Module):

    """ MBConv with expansion, plus squeeze-and-excitation and dropsample.

    Replace costly 3 X 3 convolutions with depthwise convolutions and follows a
    narrow -> wide -> narrow structure as opposed to the wide -> narrow -> wide one found in
    original residual blocks.

    Quoting original paper Section 4: "Its main building block is mobile inverted bottleneck
    MBConv (Sandler et al., 2018; Tan et al., 2019), to which we also add squeeze-and-excitation
    optimization (Hu et al., 2018)."
    """
    def __init__(
        self, n_in: int, n_out: int, expand_factor: int, kernel_size: int = 3, stride: int = 1,
        r: int = 24, p: float = 0
    ) -> None:
        super().__init__()

        expanded = expand_factor * n_in
        padding  = (kernel_size - 1) // 2
        depthwise_conv_params = {'kernel_size': kernel_size, 'padding': padding,
                                 'stride': stride, 'groups': expanded}

        self.skip_connection = (n_in == n_out) and (stride == 1)
        self.p = p

        if expand_factor == 1:
            self.expand_pw = nn.Identity()
        else:
            self.expand_pw = ConvBnAct(n_in, expanded, kernel_size=1)
        self.depthwise  = ConvBnAct(expanded, expanded, **depthwise_conv_params)
        self.se         = SEBlock(expanded, r=r)
        self.reduce_pw  = ConvBnAct(expanded, n_out, kernel_size=1, act=False)
        # self.dropsample = DropSample2d(p)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.reduce_pw(self.se(self.depthwise(self.expand_pw(x))))
        if self.skip_connection:
            if self.p > 0:
                x = drop_connect(x, self.p, self.training)
            x = x + residual
        return x


# +------------------------------------------------------------------------------------------+ #
# |                                       EFFICIENT NET                                      | #
# +------------------------------------------------------------------------------------------+ #

class EfficientNet(nn.Module):

    """ Generic EfficientNet that takes in the width and depth scale factors and scales accordingly.
    """

    def __init__(
        self, in_channels: int = 3, width_factor: float = 1., depth_factor: float = 1.,
        dropout_rate: float = 0.1, num_classes: int = 100
    ) -> None:
        super().__init__()

        base_widths   = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        base_depths   = [1, 2, 2, 3, 3, 4, 1]
        kernel_sizes  = [3, 3, 5, 3, 5, 5, 3]
        strides       = [1, 2, 2, 2, 1, 2, 1]
        drop_probas   = [0, 0.029, 0.057, 0.086, 0.114, 0.143, 0.171]
        reduce        = [4] + 6 * [24]
        expands       = [1] + 6 * [6]

        scaled_widths = [self.__scale_width(w, width_factor) for w in base_widths]
        scaled_depths = [math.ceil(depth_factor * d) for d in base_depths]

        stage_params  = [(scaled_widths[i], scaled_widths[i + 1], scaled_depths[i],
                          expands[i], kernel_sizes[i], strides[i],
                          reduce[i], drop_probas[i]) for i in range(7)]

        # Stage 1 in the original paper Table 1
        self.stem = ConvBnAct(in_channels, scaled_widths[0], stride=2, padding=1)

        # Stages 2 to 7 in the original paper Table 1
        self.stages = nn.ModuleList([self.__create_stage(*stage_params[i]) for i in range(7)])

        # Stage 9 in the original paper Table 1
        self.pre_head = nn.Sequential(
            ConvBnAct(scaled_widths[-2], scaled_widths[-1], kernel_size=1),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(dropout_rate),
            nn.Flatten(),
        )

        self.head = nn.Linear(scaled_widths[-1], num_classes)

        self.latent_dim = scaled_widths[-1]

    @staticmethod
    def __create_stage(n_in, n_out, num_layers, expand, kernel_size, stride, r, p):
        """ Creates a Sequential of MBConv. """
        common_params = {'kernel_size': kernel_size, 'r': r, 'p': p}
        layers = [MBConv(n_in, n_out, expand, stride=stride, **common_params)]
        layers += [MBConv(n_out, n_out, expand, **common_params) for _ in range(num_layers - 1)]
        return nn.Sequential(*layers)

    @staticmethod
    def __scale_width(w: int, w_factor: float) -> int:
        """ Scales width given a scale factor.
        See:
        https://stackoverflow.com/questions/60583868/how-is-the-number-of-channels-adjusted-in-efficientnet.
        """
        w *= w_factor
        new_w = (int(w + 4) // 8) * 8
        new_w = max(8, new_w)
        if new_w < 0.9 * w:
            new_w += 8
        return int(new_w)

    def extract_features(self, x: Tensor, return_all_features: bool = False) -> Tensor:
        f = self.stem(x)
        all_features = [f]
        for stage in self.stages:
            f = stage(f)
            all_features.append(f)
        f = self.pre_head(f)
        all_features.append(f)
        return all_features[-1] if not return_all_features else all_features

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.extract_features(x))

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_name(cls, network_name: str, **kwargs) -> EfficientNet:
        width, depth, _, dropout = efficientnet_params(network_name)
        scale_params = dict(width_factor=width, depth_factor=depth, dropout_rate=dropout)
        return cls(**{**scale_params, **kwargs})


def efficientnet_params(model_name: str) -> tuple[float, float, int, float]:
    """ Map EfficientNet model name to parameter coefficients.
    Official tensorflow code:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_builder.py#L35

    Args:
        model_name (str): Model name to be queried.

    Returns:
        params_dict[model_name]: A (width, depth, res, dropout) tuple.
    """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
        'efficientnet-b8': (2.2, 3.6, 672, 0.5),
        'efficientnet-l2': (4.3, 5.3, 800, 0.5),
    }
    return params_dict[model_name]
