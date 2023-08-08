import ivy
import ivy_models
from .layers import BlockParams, SimpleStemIN, ResBottleneckBlock, AnyStage
from typing import Optional, Callable

from collections import OrderedDict


class RegNet(ivy.Module):
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
        super().__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = ivy.BatchNorm2D
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = ivy.ReLU

        self.stem = stem_type(
            3,  # width_in
            stem_width,
            norm_layer,
            activation,
        )

        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
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
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = ivy.Sequential(OrderedDict(blocks))

        self.avgpool = ivy.AdaptiveAvgPool2d((1, 1))
        self.fc = ivy.Linear(current_width, num_classes)

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


def regnet_y_400mf(pretrained=True):
    """RegNet-Y-400MF model"""
    model = RegNet
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


def regnet_y_800mf(pretrained=True):
    """RegNet-Y-800MF model"""
    model = RegNet
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
