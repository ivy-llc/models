# inspired heavily by pytorch's efficient -
# https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

import ivy
import ivy_models
import builtins
import copy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Union, Tuple
from ivy_models.base import BaseSpec, BaseModel

from ivy_models.efficientnet.layers import (
    _make_divisible,
    EfficientNetConv2dNormActivation,
    EfficientNetSqueezeExcitation,
    EfficientNetStochasticDepth,
)


@dataclass
class _MBConvConfig:
    expand_ratio: float
    kernel: int
    stride: int
    input_channels: int
    out_channels: int
    num_layers: int
    block: Callable[..., ivy.Module]

    @staticmethod
    def adjust_channels(a, b, c):
        return _make_divisible(a * b, c)


class MBConvConfig(_MBConvConfig):
    # Stores information listed at Table 1 of the EfficientNet paper &
    # Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        block: Optional[Callable[..., ivy.Module]] = None,
    ) -> None:
        input_channels = self.adjust_channels(input_channels, width_mult, 8)
        out_channels = self.adjust_channels(out_channels, width_mult, 8)
        num_layers = self.adjust_depth(num_layers, depth_mult)
        if block is None:
            block = EfficientNetMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )

    @staticmethod
    def adjust_depth(a, b):
        return int(ivy.ceil(a * b))


class FusedMBConvConfig(_MBConvConfig):
    # Stores information listed at Table 4 of the EfficientNetV2 paper
    def __init__(
        self,
        expand_ratio: float,
        kernel: int,
        stride: int,
        input_channels: int,
        out_channels: int,
        num_layers: int,
        block: Optional[Callable[..., ivy.Module]] = None,
    ) -> None:
        if block is None:
            block = EfficientNetFusedMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class EfficientNetMBConv(ivy.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., ivy.Module],
        se_layer: Callable[..., ivy.Module] = EfficientNetSqueezeExcitation,
    ) -> None:
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers = []
        activation_layer = ivy.SiLU

        # expand
        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio, 8)
        if expanded_channels != cnf.input_channels:
            layers.append(
                EfficientNetConv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            EfficientNetConv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=cnf.kernel,
                stride=cnf.stride,
                groups=expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                depthwise=True,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, cnf.input_channels // 4)
        layers.append(
            se_layer(expanded_channels, squeeze_channels, activation=ivy.SiLU)
        )

        # project
        layers.append(
            EfficientNetConv2dNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = ivy.Sequential(*layers)
        self.stochastic_depth = EfficientNetStochasticDepth(stochastic_depth_prob)
        self.out_channels = cnf.out_channels

        super().__init__()

    def _forward(self, input: ivy.Array) -> ivy.Array:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNetFusedMBConv(ivy.Module):
    def __init__(
        self,
        cnf: FusedMBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., ivy.Module],
    ) -> None:
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = (
            cnf.stride == 1 and cnf.input_channels == cnf.out_channels
        )

        layers = []
        activation_layer = ivy.SiLU

        expanded_channels = cnf.adjust_channels(cnf.input_channels, cnf.expand_ratio, 8)
        if expanded_channels != cnf.input_channels:
            # fused expand
            layers.append(
                EfficientNetConv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                EfficientNetConv2dNormActivation(
                    expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
        else:
            layers.append(
                EfficientNetConv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = ivy.Sequential(*layers)
        self.stochastic_depth = EfficientNetStochasticDepth(stochastic_depth_prob)
        self.out_channels = cnf.out_channels

        super().__init__()

    def _forward(self, input: ivy.Array) -> ivy.Array:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNetSpec(BaseSpec):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., ivy.Module]] = None,
        last_channel: Optional[int] = None,
        data_format: str = "NHWC",
    ):
        super(EfficientNetSpec, self).__init__(
            inverted_residual_setting=inverted_residual_setting,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            num_classes=num_classes,
            norm_layer=norm_layer,
            last_channel=last_channel,
            data_format=data_format,
        )


class EfficientNet(BaseModel):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., ivy.Module]] = None,
        last_channel: Optional[int] = None,
        data_format: str = "NHWC",
        spec=None,
        v=None,
    ) -> None:
        """
        Efficientnet V1 and V2 main class

        Args:
        ----
            inverted_residual_setting
                (Sequence[Union[MBConvConfig, FusedMBConvConfig]]):Network structure
            dropout (float): The droupout probability
            stochastic_depth_prob (float): The stochastic depth probability
            num_classes (int): Number of classes
            norm_layer (Optional[Callable[..., nn.Module]]):
                Module specifying the normalization layer to use
            last_channel (int): The number of channels on the penultimate layer
        """
        self.spec = (
            spec
            if spec and isinstance(spec, EfficientNetSpec)
            else EfficientNetSpec(
                inverted_residual_setting,
                dropout,
                stochastic_depth_prob,
                num_classes,
                norm_layer,
                last_channel,
                data_format=data_format,
            )
        )

        super(EfficientNet, self).__init__(v=v)

    def _build(self, *args, **kwargs) -> bool:
        if self.spec.norm_layer is None:
            norm_layer = ivy.BatchNorm2D

        layers = []

        # building first layer
        firstconv_output_channels = self.spec.inverted_residual_setting[
            0
        ].input_channels
        layers.append(
            EfficientNetConv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=ivy.SiLU,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(
            cnf.num_layers for cnf in self.spec.inverted_residual_setting
        )
        stage_block_id = 0
        for cnf in self.spec.inverted_residual_setting:
            stage = []
            for _ in range(cnf.num_layers):
                # copy to avoid modifications. shallow copy is enough
                block_cnf = copy.copy(cnf)

                # overwrite info if not the first conv in the stage
                if stage:
                    block_cnf.input_channels = block_cnf.out_channels
                    block_cnf.stride = 1

                # adjust stochastic depth prob based on the depth of the stage block
                sd_prob = (
                    self.spec.stochastic_depth_prob
                    * float(stage_block_id)
                    / total_stage_blocks
                )

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(ivy.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = self.spec.inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            self.spec.last_channel
            if self.spec.last_channel is not None
            else 4 * lastconv_input_channels
        )
        layers.append(
            EfficientNetConv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=ivy.SiLU,
            )
        )

        self.features = ivy.Sequential(*layers)
        self.classifier = ivy.Sequential(
            ivy.Dropout(self.spec.dropout),
            ivy.Linear(lastconv_output_channels, self.spec.num_classes),
        )

    @classmethod
    def get_spec_class(self):
        return EfficientNetSpec

    def _forward_impl(self, x: ivy.Array, data_format=None) -> ivy.Array:
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NCHW":
            x = ivy.permute_dims(x, (0, 2, 3, 1))
        x = self.features(x)

        x = ivy.permute_dims(x, (0, 3, 1, 2))
        x = ivy.adaptive_avg_pool2d(x, 1)
        x = ivy.permute_dims(x, (0, 2, 3, 1))

        x = ivy.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

    def _forward(self, x: ivy.Array, data_format=None) -> ivy.Array:
        return self._forward_impl(x, data_format=data_format)


def _efficientnet_conf(
    arch: str,
) -> Tuple[Sequence[Union[MBConvConfig, FusedMBConvConfig]], Optional[int]]:
    inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]]
    if arch.startswith("efficientnet_b"):
        norm_layer = None
        if arch.endswith("0"):
            width_mult = 1.0
            depth_mult = 1.0
            dropout = 0.2
        elif arch.endswith("1"):
            width_mult = 1.0
            depth_mult = 1.1
            dropout = 0.2
        elif arch.endswith("2"):
            width_mult = 1.1
            depth_mult = 1.2
            dropout = 0.3
        elif arch.endswith("3"):
            width_mult = 1.2
            depth_mult = 1.4
            dropout = 0.3
        elif arch.endswith("4"):
            width_mult = 1.4
            depth_mult = 1.8
            dropout = 0.4
        elif arch.endswith("5"):
            width_mult = 1.6
            depth_mult = 2.2
            dropout = 0.4
            norm_layer = partial(ivy.BatchNorm2D, eps=0.001, momentum=0.01)
        elif arch.endswith("6"):
            width_mult = 1.8
            depth_mult = 2.6
            dropout = 0.5
            norm_layer = partial(ivy.BatchNorm2D, eps=0.001, momentum=0.01)
        elif arch.endswith("7"):
            width_mult = 2.0
            depth_mult = 3.1
            dropout = 0.5
            norm_layer = partial(ivy.BatchNorm2D, eps=0.001, momentum=0.01)
        else:
            raise ValueError(f"Unsupported model type {arch}")
        bneck_conf = partial(MBConvConfig, width_mult=width_mult, depth_mult=depth_mult)
        inverted_residual_setting = [
            bneck_conf(1, 3, 1, 32, 16, 1),
            bneck_conf(6, 3, 2, 16, 24, 2),
            bneck_conf(6, 5, 2, 24, 40, 2),
            bneck_conf(6, 3, 2, 40, 80, 3),
            bneck_conf(6, 5, 1, 80, 112, 3),
            bneck_conf(6, 5, 2, 112, 192, 4),
            bneck_conf(6, 3, 1, 192, 320, 1),
        ]
        last_channel = None
    elif arch.startswith("efficientnet_v2_s"):
        dropout = 0.2
        norm_layer = partial(ivy.BatchNorm2D, eps=0.001)
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 2),
            FusedMBConvConfig(4, 3, 2, 24, 48, 4),
            FusedMBConvConfig(4, 3, 2, 48, 64, 4),
            MBConvConfig(4, 3, 2, 64, 128, 6),
            MBConvConfig(6, 3, 1, 128, 160, 9),
            MBConvConfig(6, 3, 2, 160, 256, 15),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_m"):
        dropout = 0.3
        norm_layer = partial(ivy.BatchNorm2D, eps=0.001)
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 24, 24, 3),
            FusedMBConvConfig(4, 3, 2, 24, 48, 5),
            FusedMBConvConfig(4, 3, 2, 48, 80, 5),
            MBConvConfig(4, 3, 2, 80, 160, 7),
            MBConvConfig(6, 3, 1, 160, 176, 14),
            MBConvConfig(6, 3, 2, 176, 304, 18),
            MBConvConfig(6, 3, 1, 304, 512, 5),
        ]
        last_channel = 1280
    elif arch.startswith("efficientnet_v2_l"):
        dropout = 0.4
        norm_layer = partial(ivy.BatchNorm2D, eps=0.001)
        inverted_residual_setting = [
            FusedMBConvConfig(1, 3, 1, 32, 32, 4),
            FusedMBConvConfig(4, 3, 2, 32, 64, 7),
            FusedMBConvConfig(4, 3, 2, 64, 96, 7),
            MBConvConfig(4, 3, 2, 96, 192, 10),
            MBConvConfig(6, 3, 1, 192, 224, 19),
            MBConvConfig(6, 3, 2, 224, 384, 25),
            MBConvConfig(6, 3, 1, 384, 640, 7),
        ]
        last_channel = 1280
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel, dropout, norm_layer


def _efficient_net_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key
    W_KEY = [
        "1/0/block/0/0/weight",
        "block/1/0/weight",
    ]
    W_KEY2 = [
        "fc1/weight",
        "fc2/weight",
        "0/weight",
    ]
    if builtins.any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b 1 h w-> h w b"}
    elif "fc1/bias" in old_key or "fc2/bias" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "h-> 1 1 1 h"}
    elif builtins.any([kc in old_key for kc in W_KEY2]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w-> h w c b"}
    return new_mapping


def efficientnet_b0(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_b0"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = (
            "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth"
        )
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def efficientnet_b1(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_b1"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = (
            "https://download.pytorch.org/models/efficientnet_b1_rwightman-533bc792.pth"
        )
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def efficientnet_b2(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_b2"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = (
            "https://download.pytorch.org/models/efficientnet_b2_rwightman-bcdf34b7.pth"
        )
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def efficientnet_b3(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_b3"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = (
            "https://download.pytorch.org/models/efficientnet_b3_rwightman-cf984f9c.pth"
        )
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def efficientnet_b4(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_b4"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = (
            "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth"
        )
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def efficientnet_b5(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_b5"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = (
            "https://download.pytorch.org/models/efficientnet_b5_lukemelas-b6417697.pth"
        )
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def efficientnet_b6(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_b6"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = (
            "https://download.pytorch.org/models/efficientnet_b6_lukemelas-c76e70fd.pth"
        )
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def efficientnet_b7(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_b7"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = (
            "https://download.pytorch.org/models/efficientnet_b7_lukemelas-dcc49843.pth"
        )
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def efficientnet_v2_s(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_v2_s"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = "https://download.pytorch.org/models/efficientnet_v2_s-dd5fe13b.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def efficientnet_v2_m(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_v2_m"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = "https://download.pytorch.org/models/efficientnet_v2_m-dc08266a.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def efficientnet_v2_l(pretrained=True, data_format="NHWC"):
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_v2_l"
    )
    model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )
    if pretrained:
        url = "https://download.pytorch.org/models/efficientnet_v2_l-59c71312.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_efficient_net_torch_weights_mapping,
        )
        model.v = w_clean

    return model
