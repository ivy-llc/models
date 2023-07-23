# global
from typing import List, Optional, Type, Union
import builtins

import ivy
import ivy_models
from typing import Optional
import ivy
from typing import Any, Callable, List, Optional, Tuple
import warnings
from collections import namedtuple
from functools import partial
from ivy import Array
from ivy.stateful.module import Module

class GoogLeNet(ivy.Module):

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        init_weights: Optional[bool] = None,
        blocks: Optional[List[Callable[..., ivy.Module]]] = None,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
    ) -> None:
        super().__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of GoogleNet will be changed in future releases of "
                "ivyvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        if len(blocks) != 3:
            raise ValueError(f"blocks length should be 3 instead of {len(blocks)}")
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits

        self.conv1 = conv_block(3, 64, 7, 2, 3)
        self.maxpool1 = ivy.MaxPool2D(3, stride=2)
        self.conv2 = conv_block(64, 64, 1)
        self.conv3 = conv_block(64, 192, 3, 1, 1)
        self.maxpool2 = ivy.MaxPool2D(3, stride=2)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = ivy.MaxPool2D(3, stride=2)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = ivy.MaxPool2D(2, stride=2)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None  # type: ignore[assignment]
            self.aux2 = None  # type: ignore[assignment]

        self.avgpool = ivy.AdaptiveAvgPool2d((1, 1))
        self.dropout = ivy.Dropout(p=dropout)
        self.fc = ivy.Linear(1024, num_classes)

        if init_weights:
            for m in self.modules():
                if isinstance(m, ivy.Conv2D) or isinstance(m, ivy.Linear):
                    ivy.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                elif isinstance(m, ivy.BatchNorm2d):
                    ivy.init.constant_(m.weight, 1)
                    ivy.init.constant_(m.bias, 0)


    def _forward(self, x: Array) -> Tuple[Array, Optional[Array], Optional[Array]]:
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux1: Optional[Array] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        aux2: Optional[Array] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = ivy.flatten(x)
        # N x 1024
        x = self.dropout(x)
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux2, aux1

   

class Inception(ivy.Module):
    def __init__(
        self,
        in_chaivyels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., ivy.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_chaivyels, ch1x1, 1)

        self.branch2 = ivy.Sequential(
            conv_block(in_chaivyels, ch3x3red, 1),
              conv_block(ch3x3red, ch3x3, 3, 1, 1)
        )

        self.branch3 = ivy.Sequential(
            conv_block(in_chaivyels, ch5x5red, 1),
            # Here, kernel_size=3 instead of kernel_size=5 is a known bug.
            # Please see https://github.com/pyivy/vision/issues/906 for details.
            conv_block(ch5x5red, ch5x5, 3, 1, 1),
        )

        self.branch4 = ivy.Sequential(
            ivy.MaxPool2D(kernel_size=3, stride=1, padding=1),
            conv_block(in_chaivyels, pool_proj, 1),
        )

    def _forward(self, x: Array) -> List[Array]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Array) -> Array:
        outputs = self._forward(x)
        return ivy.cat(outputs, 1)


class InceptionAux(ivy.Module):
    def __init__(
        self,
        in_chaivyels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., ivy.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_chaivyels, 128, 1)

        self.fc1 = ivy.Linear(2048, 1024)
        self.fc2 = ivy.Linear(1024, num_classes)
        self.dropout = ivy.Dropout(p=dropout)

    def _forward(self, x: Array) -> Array:
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = ivy.permute_dims(x, (0, 3, 1, 2))
        x = ivy.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.conv(x)
        # N x 128 x 4 x 4
        x = ivy.flatten(x)
        # N x 2048
        x = ivy.relu(self.fc1(x), inplace=True)
        # N x 1024
        x = self.dropout(x)
        # N x 1024
        x = self.fc2(x)
        # N x 1000 (num_classes)

        return x


class BasicConv2d(ivy.Module):
    def __init__(self, in_chaivyels: int, out_chaivyels: int, *args: Any) -> None:
        super().__init__()
        self.conv = ivy.Conv2D(in_chaivyels, out_chaivyels, with_bias=False, *args)
        self.bn = ivy.BatchNorm2d(out_chaivyels, eps=0.001)

    def _forward(self, x: Array) -> Array:
        x = self.conv(x)
        x = self.bn(x)
        return ivy.relu(x, inplace=True)
    


