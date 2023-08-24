from typing import Callable, List, Optional, Sequence, Tuple, Union, Any

import ivy

import math
import collections
from collections import OrderedDict
from itertools import repeat


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.
    reference: https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/utils.py#L8 # noqa: E501

    Args:
    ----
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    A function taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvNormActivation(ivy.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        norm_layer: Optional[Callable[..., ivy.Module]] = ivy.BatchNorm2D,
        activation_layer: Optional[Callable[..., ivy.Module]] = ivy.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = True,
        conv_layer: Callable[..., ivy.Module] = ivy.Conv2D,
    ) -> None:
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = (
                    len(kernel_size)
                    if isinstance(kernel_size, Sequence)
                    else len(dilation)
                )
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple(
                    (kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim)
                )
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                [kernel_size],
                stride,
                padding,
                dilations=dilation,
                with_bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        super().__init__(*layers)

        self.out_channels = out_channels

        if self.__class__ == ConvNormActivation:
            ivy.warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation instead."  # noqa: E501
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
    ----
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by
            the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all
            four sides of the input. Default: None, in which case it will be
            calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        norm_layer (Callable[..., ivy.Module], optional): Norm layer that will
            be stacked on top of the convolution layer.If ``None`` this
            layer won't be used. Default: ``ivy.BatchNorm2D``
        activation_layer (Callable[..., ivy.Module], optional): Activation function
            which will be stacked on top of the normalization layer (if not None),
            otherwise on top of the conv layer. If ``None`` this layer won't be
            used. Default: ``ivy.ReLU``
        dilation (int): Spacing between kernel elements.
            Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally
            do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By
            default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        norm_layer: Optional[Callable[..., ivy.Module]] = ivy.BatchNorm2D,
        activation_layer: Optional[Callable[..., ivy.Module]] = ivy.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            ivy.Conv2D,
        )


class SqueezeExcitation(ivy.Module):
    """
    The Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1). # noqa: E501
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3. # noqa: E501

    Args:
    ----
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., ivy.Module], optional): ``delta`` activation. Default: ``ivy.ReLU`` # noqa: E501
        scale_activation (Callable[..., ivy.Module]): ``sigma`` activation. Default: ``ivy.Sigmoid`` # noqa: E501
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., ivy.Module] = ivy.ReLU,
        scale_activation: Callable[..., ivy.Module] = ivy.sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = ivy.AdaptiveAvgPool2d(1)
        self.fc1 = ivy.Conv2D(input_channels, squeeze_channels, 1, strides=1, padding=0)
        self.fc2 = ivy.Conv2D(squeeze_channels, input_channels, 1, strides=1, padding=0)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: ivy.Array) -> ivy.Array:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: ivy.Array) -> ivy.Array:
        scale = self._scale(input)
        return scale * input


class SimpleStemIN(Conv2dNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., ivy.Module],
        activation_layer: Callable[..., ivy.Module],
    ) -> None:
        super().__init__(
            width_in,
            width_out,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )


class BottleneckTransform(ivy.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., ivy.Module],
        activation_layer: Callable[..., ivy.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        w_b // group_width

        layers["a"] = Conv2dNormActivation(
            width_in,
            w_b,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        layers["b"] = Conv2dNormActivation(
            w_b,
            w_b,
            kernel_size=3,
            stride=stride,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = Conv2dNormActivation(
            w_b,
            width_out,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=None,
        )
        super().__init__(layers)


class ResBottleneckBlock(ivy.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., ivy.Module],
        activation_layer: Callable[..., ivy.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        self.width_in = width_in
        self.width_out = width_out
        self.stride = stride
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.group_width = group_width
        self.bottleneck_multiplier = bottleneck_multiplier
        self.se_ratio = se_ratio
        # Use skip connection with projection if shape changes
        self.proj = None
        super().__init__()

    def _build(self, *args, **kwargs):
        should_proj = (self.width_in != self.width_out) or (self.stride != 1)
        if should_proj:
            self.proj = Conv2dNormActivation(
                self.width_in,
                self.width_out,
                kernel_size=1,
                stride=self.stride,
                norm_layer=self.norm_layer,
                activation_layer=None,
            )
        self.f = BottleneckTransform(
            self.width_in,
            self.width_out,
            self.stride,
            self.norm_layer,
            self.activation_layer,
            self.group_width,
            self.bottleneck_multiplier,
            self.se_ratio,
        )

    def _forward(self, x: ivy.Array) -> ivy.Array:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(ivy.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., ivy.Module],
        norm_layer: Callable[..., ivy.Module],
        activation_layer: Callable[..., ivy.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()

        self.blocks = OrderedDict()
        for i in range(depth):
            block = block_constructor(  # noqa: F841
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )
            self.blocks[f"block{stage_index}-{i}"] = block


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
        """
        Programmatically compute all the per-block settings,
        given the RegNet parameters.

        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space

        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.

        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage.
        """
        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")
        # Compute the block widths. Each stage has one unique block width
        widths_cont = ivy.arange(depth) * w_a + w_0
        block_capacity = ivy.round(ivy.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (
            (ivy.round(ivy.divide(w_0 * ivy.pow(w_m, block_capacity), QUANT)) * QUANT)
            .int()
            .tolist()
        )
        num_stages = len(set(block_widths))

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = (
            ivy.diff(ivy.array([d for d, t in enumerate(splits) if t])).int().tolist()
        )

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(
            self.widths,
            self.strides,
            self.depths,
            self.group_widths,
            self.bottleneck_multipliers,
        )

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        """
        Adjusts the compatibility of widths and groups,
        depending on the bottleneck ratio.
        """
        # Compute all widths for the current settings
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Compute the adjusted widths so that stage and group widths fit
        ws_bot = [
            _make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)
        ]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min
