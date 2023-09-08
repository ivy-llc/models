# global
from typing import List, Optional, Type, Union

import ivy
from ivy_models.Regnet.layers import BasicBlock, Bottleneck
from ivy_models.base import BaseSpec


class RegNetSpec(BaseSpec):
    """RegNetSpec class"""

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        base_width: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super(RegNetSpec, self).__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            base_width=base_width,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )


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
        _log_api_usage_once(self)

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = ivy.BatchNorm2d
        if block_type is None:
            block_type = RegBottleneckBlock
        if activation is None:
            activation = ivy.ReLU

        # Ad hoc stem
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
        self.fc = ivy.Linear(in_featuReg=current_width, out_featuReg=num_classes)

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

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
