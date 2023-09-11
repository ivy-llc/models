# global
import math
from collections import OrderedDict
from typing import Callable, Optional, Type
import builtins
import ivy_models
import ivy
from ivy_models.regnet.layers import (
    SimpleStemIN,
    ResBottleneckBlock,
    AnyStage,
    BlockParams,
)
from ivy_models.base import BaseSpec
from functools import partial


class RegNetSpec(BaseSpec):
    """RegNetSpec class"""

    def __init__(
        self,
        block_params: Type[BlockParams],
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., ivy.Module]] = None,
        block_type: Optional[Callable[..., ivy.Module]] = None,
        norm_layer: Optional[Callable[..., ivy.Module]] = None,
        activation: Optional[Callable[..., ivy.Module]] = None,
    ) -> None:
        super(RegNetSpec, self).__init__(
            block_params=block_params,
            num_classes=num_classes,
            stem_width=stem_width,
            stem_type=stem_type,
            block_type=block_type,
            norm_layer=norm_layer,
            activation=activation,
        )


ivy.conv2d


class RegNet(ivy.Module):
    """
    Residual Neural Network (ResNet) architecture.

    Args::
        block (Type[Union[BasicBlock, Bottleneck]]):
            The block type used in the ResNet architecture.
        layers: List of integers specifying the number of blocks in each layer.
        num_classes (int): Number of output classes. Defaults to 1000.
        base_width (int): The base width of the ResNet. Defaults to 64.
        replace_stride_with_dilation (Optional[List[bool]]):
            List indicating whether to replace stride with dilation.
        v (ivy.Container): Unused parameter. Can be ignored.

    """

    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 1000,
        stem_width: int = 32,
        stem_type: Optional[Callable[..., ivy.Module]] = None,
        block_type: Optional[Callable[..., ivy.Module]] = None,
        norm_layer: Optional[Callable[..., ivy.Module]] = None,
        activation: Optional[Callable[..., ivy.Module]] = None,
        spec=None,
        v: ivy.Container = None,
        **kwargs,
    ) -> None:
        self.spec = (
            spec
            if spec and isinstance(spec, RegNetSpec)
            else RegNetSpec(
                block_params,
                num_classes,
                stem_width,
                stem_type,
                block_type,
                norm_layer,
                activation,
            )
        )

        super(RegNet, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        norm_layer = kwargs.pop(
            "norm_layer", partial(ivy.BatchNorm2D, eps=1e-05, momentum=0.1)
        )
        if self.spec.stem_type is None:
            stem_type = SimpleStemIN
        if self.spec.norm_layer is None:
            norm_layer = ivy.BatchNorm2D
        if self.spec.block_type is None:
            block_type = ResBottleneckBlock
        if self.spec.activation is None:
            activation = ivy.ReLU

        # Ad hoc stem
        self.stem = stem_type(
            3,  # width_in
            self.spec.stem_width,
            self.spec.norm_layer,
            self.spec.activation,
        )

        current_width = self.spec.stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(self.spec.block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        self.spec.block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = ivy.Sequential(OrderedDict(blocks))

        self.avgpool = ivy.AdaptiveAvgPool2D((1, 1))
        self.fc = ivy.Linear(
            in_featuReg=current_width, out_featuReg=self.spec.num_classes
        )

        # Performs RegNet-style weight initialization
        for m in self.modules():
            if isinstance(m, ivy.Conv2D):
                # Note that there is no bias due to BN
                fan_out = m._filter_shape[0] * m._filter_shape[1] * m._output_channels
                ivy.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, ivy.BatchNorm2D):
                ivy.init.ones_(m.weight)
                ivy.init.zeros_(m.bias)
            elif isinstance(m, ivy.Linear):
                ivy.init.normal_(m.weight, mean=0.0, std=0.01)
                ivy.init.zeros_(m.bias)

    def _forward(self, x):
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


def _regnet_y_400mf_torch_weights_mapping(old_key, new_key):
    W_KEY = ["conv1/weight", "conv2/weight", "conv3/weight", "downsample/0/weight"]
    new_mapping = new_key
    if builtins.any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_mapping


def regnet_y_400mf(pretrained=True):
    """ResNet-18 model"""
    block_params = BlockParams.from_init_params(
        depth=20,
        w_0=232,
        w_a=115.89,
        w_m=2.53,
        group_width=232,
        se_ratio=0.25,
    )
    model = RegNet(
        block_params,
    )
    if pretrained:
        url = "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_regnet_y_400mf_torch_weights_mapping,
        )
        model.v = w_clean
    return model


# regnet_y_400mf = None
regnet_y_800mf = None
regnet_y_1_6gf = None
regnet_y_3_2gf = None
regnet_y_8gf = None
regnet_y_16gf = None
regnet_y_32gf = None
# regnet_y_128gf = None
regnet_x_400mf = None
regnet_x_800mf = None
regnet_x_1_6gf = None
regnet_x_3_2gf = None
regnet_x_8gf = None
regnet_x_16gf = None
regnet_x_32gf = None

VARIANTS = {
    "regnet_y_400mf": BlockParams.from_init_params(
        depth=20,
        w_0=232,
        w_a=115.89,
        w_m=2.53,
        group_width=232,
        se_ratio=0.25,
    ),
    "regnet_y_800mf": BlockParams.from_init_params(
        depth=16,
        w_0=48,
        w_a=27.89,
        w_m=2.09,
        group_width=8,
        se_ratio=0.25,
    ),
    "regnet_y_1_6gf": BlockParams.from_init_params(
        depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25
    ),
    "regnet_y_3_2gf": BlockParams.from_init_params(
        depth=21, w_0=80, w_a=42.63, w_m=2.66, group_width=24, se_ratio=0.25
    ),
    "regnet_y_8gf": BlockParams.from_init_params(
        depth=17, w_0=192, w_a=76.82, w_m=2.19, group_width=56, se_ratio=0.25
    ),
    "regnet_y_16gf": BlockParams.from_init_params(
        depth=18,
        w_0=200,
        w_a=106.23,
        w_m=2.48,
        group_width=112,
        se_ratio=0.25,
    ),
    "regnet_y_32gf": BlockParams.from_init_params(
        depth=20,
        w_0=232,
        w_a=115.89,
        w_m=2.53,
        group_width=232,
        se_ratio=0.25,
    ),
    #  "regnet_y_128gf": BlockParams.from_init_params(
    #     depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, se_ratio=0.25
    # ),
    "regnet_x_400mf": BlockParams.from_init_params(
        depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16
    ),
    "regnet_x_800mf": BlockParams.from_init_params(
        depth=16, w_0=56, w_a=35.73, w_m=2.28, group_width=16
    ),
    "regnet_x_1_6gf": BlockParams.from_init_params(
        depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24
    ),
    "regnet_x_3_2gf": BlockParams.from_init_params(
        depth=25, w_0=88, w_a=26.31, w_m=2.25, group_width=48
    ),
    "regnet_x_8gf": BlockParams.from_init_params(
        depth=23, w_0=80, w_a=49.56, w_m=2.88, group_width=120
    ),
    "regnet_x_16gf": BlockParams.from_init_params(
        depth=22, w_0=216, w_a=55.59, w_m=2.1, group_width=128
    ),
    "regnet_x_32gf": BlockParams.from_init_params(
        depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168
    ),
}
