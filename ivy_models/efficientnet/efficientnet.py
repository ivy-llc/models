# inspired heavily by pytorch's efficient -
# https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py

import ivy
import copy
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence, Union, Tuple

from misc import (
    _make_divisible,
    Conv2dNormActivation,
    SqueezeExcitation,
    StochasticDepth,
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
            block = MBConv
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
            block = FusedMBConv
        super().__init__(
            expand_ratio,
            kernel,
            stride,
            input_channels,
            out_channels,
            num_layers,
            block,
        )


class MBConv(ivy.Module):
    def __init__(
        self,
        cnf: MBConvConfig,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., ivy.Module],
        se_layer: Callable[..., ivy.Module] = SqueezeExcitation,
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
                Conv2dNormActivation(
                    cnf.input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
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
            Conv2dNormActivation(
                expanded_channels,
                cnf.out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = ivy.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)
        self.out_channels = cnf.out_channels

        super().__init__()

    def _forward(self, input: ivy.Array) -> ivy.Array:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class FusedMBConv(ivy.Module):
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
                Conv2dNormActivation(
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
                Conv2dNormActivation(
                    expanded_channels,
                    cnf.out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    cnf.input_channels,
                    cnf.out_channels,
                    kernel_size=cnf.kernel,
                    stride=cnf.stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = ivy.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob)
        self.out_channels = cnf.out_channels

        super().__init__()

    def _forward(self, input: ivy.Array) -> ivy.Array:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class EfficientNet(ivy.Module):
    def __init__(
        self,
        inverted_residual_setting: Sequence[Union[MBConvConfig, FusedMBConvConfig]],
        dropout: float,
        stochastic_depth_prob: float = 0.2,
        num_classes: int = 1000,
        norm_layer: Optional[Callable[..., ivy.Module]] = None,
        last_channel: Optional[int] = None,
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
        if norm_layer is None:
            norm_layer = ivy.BatchNorm2D

        layers = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=ivy.SiLU,
            )
        )

        # building inverted residual blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in inverted_residual_setting)
        stage_block_id = 0
        for cnf in inverted_residual_setting:
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
                    stochastic_depth_prob * float(stage_block_id) / total_stage_blocks
                )

                stage.append(block_cnf.block(block_cnf, sd_prob, norm_layer))
                stage_block_id += 1

            layers.append(ivy.Sequential(*stage))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = (
            last_channel if last_channel is not None else 4 * lastconv_input_channels
        )
        layers.append(
            Conv2dNormActivation(
                lastconv_input_channels,
                lastconv_output_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=ivy.SiLU,
            )
        )

        self.features = ivy.Sequential(*layers)
        self.classifier = ivy.Sequential(
            ivy.Dropout(dropout),
            ivy.Linear(lastconv_output_channels, num_classes),
        )

        super(EfficientNet, self).__init__(v=v)

    def _forward_impl(self, x: ivy.Array) -> ivy.Array:
        x = self.features(x)

        x = ivy.permute_dims(x, (0, 3, 1, 2))
        x = ivy.adaptive_avg_pool2d(x, 1)
        x = ivy.permute_dims(x, (0, 2, 3, 1))

        x = ivy.flatten(x, start_dim=1)
        x = self.classifier(x)

        return x

    def _forward(self, x: ivy.Array) -> ivy.Array:
        return self._forward_impl(x)


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


if __name__ == "__main__":
    # Preprocess torch image
    import torch
    from torchvision import transforms
    from PIL import Image

    ivy.set_torch_backend()

    filename = "images/dog.jpeg"
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    torch_img = Image.open(filename)
    torch_img = preprocess(torch_img)
    torch_img = torch.unsqueeze(torch_img, 0)
    img = torch_img.numpy().reshape(1, 224, 224, 3)

    # supported archs -
    # efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    # efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7,
    # efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
    inverted_residual_setting, last_channel, dropout, norm_layer = _efficientnet_conf(
        "efficientnet_b0"
    )
    ivy_model = EfficientNet(
        inverted_residual_setting,
        dropout,
        norm_layer=norm_layer,
        last_channel=last_channel,
    )

    output = ivy.softmax(ivy_model(ivy.asarray(img)))  # pass the image to the model
    print(output.shape)
    # classes = ivy.argsort(output[0], descending=True)[:3]  # get the top 3 classes
    # logits = ivy.gather(output[0], classes)  # get the logits

    # print("Indices of the top 3 classes are:", classes)
    # print("Logits of the top 3 classes are:", logits)
