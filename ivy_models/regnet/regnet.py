import ivy
import ivy_models
from ivy_models.base import BaseModel, BaseSpec
from .layers import BlockParams, SimpleStemIN, ResBottleneckBlock, AnyStage
from typing import Optional, Callable

from collections import OrderedDict


class RegNetSpec(BaseSpec):
    def __init__(
        self,
        block_params: BlockParams,
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


class RegNet(BaseModel):
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
        self.block_params = block_params
        self.num_classes = num_classes
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.block_type = block_type
        self.norm_layer = norm_layer
        self.activation = activation

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
        if self.stem_type is None:
            self.stem_type = SimpleStemIN
        if self.norm_layer is None:
            self.norm_layer = ivy.BatchNorm2D
        if self.block_type is None:
            self.block_type = ResBottleneckBlock
        if self.activation is None:
            self.activation = ivy.ReLU

        self.stem = self.stem_type(
            3,  # width_in
            self.stem_width,
            self.norm_layer,
            self.activation,
        )

        current_width = self.stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(
            zip(
                self.block_params.widths,
                self.block_params.strides,
                self.block_params.depths,
                self.block_params.group_widths,
                self.block_params.bottleneck_multipliers,
            )
        ):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        self.block_type,
                        self.norm_layer,
                        self.activation,
                        group_width,
                        bottleneck_multiplier,
                        self.block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = ivy.Sequential(OrderedDict(blocks))
        self.avgpool = ivy.AdaptiveAvgPool2d((1, 1))
        self.fc = ivy.Linear(current_width, self.num_classes)

    @classmethod
    def get_spec_class(self):
        return RegNetSpec

    def _forward(self, x: ivy.Array) -> ivy.Array:
        x = self.stem(x)
        x = self.trunk_output(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)
        return x


def _regnet_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key
    if "weight" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    elif "bias" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "h -> 1 h 1 1"}

    return new_mapping


def regnet_y_400mf(num_classes: int = 1000, stem_width: int = 32, pretrained=True):
    """RegNet-Y-400MF model"""
    model = RegNet(BlockParams, num_classes, stem_width)
    if pretrained:
        url = "https://download.pytorch.org/models/regnet_y_400mf-c65dace8.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_regnet_torch_weights_mapping,
        )
        model.v = w_clean
    return model


def regnet_y_800mf(num_classes: int = 1000, stem_width: int = 32, pretrained=True):
    """RegNet-Y-800MF model"""
    model = RegNet(BlockParams, num_classes, stem_width)
    if pretrained:
        url = "https://download.pytorch.org/models/regnet_y_400mf-e6988f5f.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_regnet_torch_weights_mapping,
        )
        model.v = w_clean
    return model
