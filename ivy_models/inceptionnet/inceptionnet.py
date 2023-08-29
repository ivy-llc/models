# inspired heavily by pytorch's efficient -
# https://github.com/pytorch/vision/blob/main/torchvision/models/inception.py

import ivy
import ivy_models
from ivy_models.inceptionnet.layers import (
    InceptionBasicConv2d,
    InceptionAux,
    InceptionA,
    InceptionB,
    InceptionC,
    InceptionD,
    InceptionE,
)
from ivy_models.base import BaseSpec, BaseModel
from typing import List


class InceptionNetSpec(BaseSpec):
    def __init__(
        self, num_classes=1000, training=False, dropout=0.5, data_format="NHWC"
    ):
        if not training:
            dropout = 0
        super(InceptionNetSpec, self).__init__(
            num_classes=num_classes,
            training=training,
            dropout=dropout,
            data_format=data_format,
        )


class InceptionV3(BaseModel):
    """An Ivy native implementation of InceptionNet"""

    """
    Args:
        ----
            num_classes (int): Number of classes
            training (bool): Set the mode for model execution
                # If training=True, the model will be in training mode
                # If training=False, the model will be in evaluation mode
            dropout (float): The droupout probability
    """

    def __init__(
        self,
        num_classes: int = 1000,
        training: bool = False,
        dropout: float = 0.5,
        data_format: str = "NHWC",
        spec=None,
        v=None,
    ) -> None:
        if not training:
            dropout = 0
        self.spec = (
            spec
            if spec and isinstance(spec, InceptionNetSpec)
            else InceptionNetSpec(
                num_classes=num_classes,
                training=training,
                dropout=dropout,
                data_format=data_format,
            )
        )
        super(InceptionV3, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        conv_block = InceptionBasicConv2d
        inception_a = InceptionA
        inception_b = InceptionB
        inception_c = InceptionC
        inception_d = InceptionD
        inception_e = InceptionE
        inception_aux = InceptionAux

        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=[3, 3], stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=[3, 3])
        self.Conv2d_2b_3x3 = conv_block(
            32, 64, kernel_size=[3, 3], padding=[[1, 1], [1, 1]]
        )
        self.maxpool1 = ivy.MaxPool2D([3, 3], 2, 0)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=[1, 1])
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=[3, 3])
        self.maxpool2 = ivy.MaxPool2D([3, 3], 2, 0)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)

        # if is used only when the model is in training mode
        if self.spec.training:
            self.AuxLogits = inception_aux(768, self.spec.num_classes)

        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.avgpool = ivy.AdaptiveAvgPool2d((1, 1))
        self.dropout = ivy.Dropout(prob=self.spec.dropout)
        self.fc = ivy.Linear(2048, self.spec.num_classes)

    @classmethod
    def get_spec_class(self):
        return InceptionNetSpec

    def _forward(self, x: ivy.Array, data_format=None) -> List[ivy.Array]:
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NCHW":
            x = ivy.permute_dims(x, (0, 2, 3, 1))

        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149

        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147

        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147

        x = self.maxpool1(x)
        # N x 64 x 73 x 73

        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73

        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71

        x = self.maxpool2(x)
        # N x 192 x 35 x 35

        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35

        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35

        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35

        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17

        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17

        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17

        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17

        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17

        aux = None
        if self.spec.training:
            aux = self.AuxLogits(x)
        #         # N x 768 x 17 x 17

        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8

        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8

        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8

        # Adaptive average pooling
        x = ivy.permute_dims(x, (0, 3, 1, 2))
        x = self.avgpool(x)
        x = ivy.permute_dims(x, (0, 2, 3, 1))

        # N x 2048 x 1 x 1

        x = self.dropout(x)
        # N x 2048 x 1 x 1

        x = ivy.flatten(x, start_dim=1)
        # N x 2048

        x = self.fc(x)
        # N x 1000 (num_classes)

        return x, aux


def _inceptionNet_v3_torch_weights_mapping(old_key, new_key):
    if "conv/weight" in old_key:
        return {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_key


def inceptionNet_v3(
    pretrained=True, training=False, num_classes=1000, dropout=0.5, data_format="NHWC"
):
    """InceptionNet-V3 model"""
    model = InceptionV3(
        num_classes=num_classes,
        training=training,
        dropout=dropout,
        data_format=data_format,
    )
    if pretrained:
        url = "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked", "AuxLogits"],
            custom_mapping=_inceptionNet_v3_torch_weights_mapping,
        )

        model.v = w_clean
    return model
