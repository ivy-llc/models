# global
from typing import List, Optional, Type, Union
import builtins

import ivy
import ivy_models
from ivy_models.resnet.layers import conv1x1, BasicBlock, Bottleneck
from ivy_models.base import BaseSpec, BaseModel


class ResNetSpec(BaseSpec):
    """ResNetSpec class"""

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        base_width: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
    ) -> None:
        super(ResNetSpec, self).__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            base_width=base_width,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )


class ResNet(BaseModel):
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
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        base_width: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        spec=None,
        v: ivy.Container = None,
    ) -> None:
        self.spec = (
            spec
            if spec and isinstance(spec, ResNetSpec)
            else ResNetSpec(
                block, layers, num_classes, base_width, replace_stride_with_dilation
            )
        )

        super(ResNet, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.inplanes = 64
        self.dilation = 1
        if self.spec.replace_stride_with_dilation is None:
            self.spec.replace_stride_with_dilation = [False, False, False]

        self.conv1 = ivy.Conv2D(3, self.inplanes, [7, 7], 2, 3, with_bias=False)
        self.bn1 = ivy.BatchNorm2D(self.inplanes, training=False)
        self.relu = ivy.ReLU()
        self.maxpool = ivy.MaxPool2D(3, 2, 1)
        self.layer1 = self._make_layer(self.spec.block, 64, self.spec.layers[0])
        self.layer2 = self._make_layer(
            self.spec.block,
            128,
            self.spec.layers[1],
            stride=2,
            dilate=self.spec.replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            self.spec.block,
            256,
            self.spec.layers[2],
            stride=2,
            dilate=self.spec.replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            self.spec.block,
            512,
            self.spec.layers[3],
            stride=2,
            dilate=self.spec.replace_stride_with_dilation[2],
        )
        self.avgpool = ivy.AdaptiveAvgPool2d((1, 1))
        self.fc = ivy.Linear(512 * self.spec.block.expansion, self.spec.num_classes)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> ivy.Sequential:
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = ivy.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                ivy.BatchNorm2D(planes * block.expansion, training=False),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.spec.base_width,
                previous_dilation,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    base_width=self.spec.base_width,
                    dilation=self.dilation,
                )
            )

        return ivy.Sequential(*layers)

    @classmethod
    def get_spec_class(self):
        return ResNetSpec

    def _forward(self, x):
        dtype = x.dtype
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = ivy.asarray(x, dtype=dtype)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = ivy.permute_dims(x, (0, 3, 1, 2))
        x = self.avgpool(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)
        return x


def _resnet_torch_weights_mapping(old_key, new_key):
    W_KEY = ["conv1/weight", "conv2/weight", "conv3/weight", "downsample/0/weight"]
    new_mapping = new_key
    if builtins.any([kc in old_key for kc in W_KEY]):
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_mapping


def resnet_18(pretrained=True):
    """ResNet-18 model"""
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        url = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_resnet_torch_weights_mapping,
        )
        model.v = w_clean
    return model


def resnet_34(pretrained=True):
    """ResNet-34 model"""
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        url = "https://download.pytorch.org/models/resnet34-333f7ec4.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_resnet_torch_weights_mapping,
        )
        model.v = w_clean
    return model


def resnet_50(pretrained=True):
    """ResNet-50 model"""
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_resnet_torch_weights_mapping,
        )
        model.v = w_clean
    return model


def resnet_101(pretrained=True):
    """ResNet-101 model"""
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        url = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_resnet_torch_weights_mapping,
        )
        model.v = w_clean
    return model


def resnet_152(pretrained=True):
    """ResNet-152 model"""
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    if pretrained:
        url = "https://download.pytorch.org/models/resnet152-f82ba261.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_resnet_torch_weights_mapping,
        )
        model.v = w_clean
    return model
