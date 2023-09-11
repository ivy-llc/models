# global
import math
from collections import OrderedDict
from typing import Any, Callable, Optional, Type, Union, List
import builtins
import ivy_models
import ivy
from ivy_models.regnet.layers import (
    SimpleStemIN,
    BottleneckTransform,
    ResBottleneckBlock,
    AnyStage,
    BlockParams,
    RegBottleneckBlock,
)
from ivy_models.base import BaseSpec


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
        if self.spec.stem_type is None:
            stem_type = SimpleStemIN
        if self.spec.norm_layer is None:
            norm_layer = ivy.BatchNorm2d
        if self.spec.block_type is None:
            block_type = RegBottleneckBlock
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

        self.avgpool = ivy.AdaptiveAvgPool2d((1, 1))
        self.fc = ivy.Linear(
            in_featuReg=current_width, out_featuReg=self.spec.num_classes
        )

        # Performs RegNet-style weight initialization
        for m in self.modules():
            if isinstance(m, ivy.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                ivy.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, ivy.BatchNorm2d):
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


def _RegNet_Y_400MF_torch_weights_mapping(old_key, new_key):
    W_KEY = ["conv1/weight", "conv2/weight", "conv3/weight", "downsample/0/weight"]
    new_mapping = new_key
    if builtins.any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_mapping


def RegNet_Y_400MF(pretrained=True):
    """ResNet-18 model"""
    model = RegNet()
    if pretrained:
        url = "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_RegNet_Y_400MF_torch_weights_mapping,
        )
        model.v = w_clean
    return model
