# global
import ivy
import ivy_models
from ivy_models.base import BaseSpec, BaseModel
from ivy_models.googlenet.layers import (
    InceptionConvBlock,
    InceptionBlock,
    InceptionAuxiliaryBlock,
)


class GoogLeNetSpec(BaseSpec):
    def __init__(
        self,
        training=False,
        num_classes=1000,
        dropout=0.4,
        aux_dropout=0.7,
        data_format="NCHW",
    ):
        if not training:
            dropout = 0
            aux_dropout = 0
        super(GoogLeNetSpec, self).__init__(
            num_classes=num_classes,
            dropout=dropout,
            aux_dropout=aux_dropout,
            data_format=data_format,
        )


class GoogLeNet(BaseModel):
    def __init__(
        self,
        training=False,
        num_classes=1000,
        dropout=0.4,
        aux_dropout=0.7,
        data_format="NCHW",
        spec=None,
        v=None,
    ):
        self.spec = (
            spec
            if spec and isinstance(spec, GoogLeNetSpec)
            else GoogLeNetSpec(
                training=training,
                num_classes=num_classes,
                dropout=dropout,
                aux_dropout=aux_dropout,
                data_format=data_format,
            )
        )
        super(GoogLeNet, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.conv1 = InceptionConvBlock(3, 64, [7, 7], 2, padding=3)

        self.conv2 = InceptionConvBlock(64, 64, [1, 1], 1, padding=0)
        self.conv3 = InceptionConvBlock(64, 192, [3, 3], 1, padding=1)

        self.inception3A = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3B = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.inception4A = InceptionBlock(480, 192, 96, 208, 16, 48, 64)

        self.aux4A = InceptionAuxiliaryBlock(
            512, self.spec.num_classes, self.spec.aux_dropout
        )

        self.inception4B = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4C = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4D = InceptionBlock(512, 112, 144, 288, 32, 64, 64)

        self.aux4D = InceptionAuxiliaryBlock(
            528, self.spec.num_classes, self.spec.aux_dropout
        )

        self.inception4E = InceptionBlock(528, 256, 160, 320, 32, 128, 128)

        self.inception5A = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5B = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.pool6 = ivy.AdaptiveAvgPool2d([1, 1])

        self.dropout = ivy.Dropout(self.spec.dropout)
        self.fc = ivy.Linear(1024, self.spec.num_classes, with_bias=False)

    @classmethod
    def get_spec_class(self):
        return GoogLeNetSpec

    def _forward(self, x, data_format=None):
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NHWC":
            x = ivy.permute_dims(x, (0, 3, 1, 2))

        out = self.conv1(x)
        out = ivy.max_pool2d(out, [3, 3], 2, 0, ceil_mode=True, data_format="NCHW")
        out = self.conv2(out)
        out = self.conv3(out)
        out = ivy.max_pool2d(out, [3, 3], 2, 0, ceil_mode=True, data_format="NCHW")
        out = self.inception3A(out)
        out = self.inception3B(out)
        out = ivy.max_pool2d(out, [3, 3], 2, 0, ceil_mode=True, data_format="NCHW")
        out = self.inception4A(out)

        aux1 = self.aux4A(out)

        out = self.inception4B(out)
        out = self.inception4C(out)
        out = self.inception4D(out)

        aux2 = self.aux4D(out)

        out = self.inception4E(out)
        out = ivy.max_pool2d(out, [2, 2], 2, 0, ceil_mode=True, data_format="NCHW")
        out = self.inception5A(out)
        out = self.inception5B(out)
        out = self.pool6(out)
        out = ivy.flatten(out, start_dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out, aux1, aux2


def _inceptionNet_torch_weights_mapping(old_key, new_key):
    if "conv/weight" in old_key:
        return {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_key


def inceptionNet_v1(
    pretrained=True,
    training=False,
    num_classes=1000,
    dropout=0.4,
    aux_dropout=0.7,
    data_format="NCHW",
):
    """InceptionNet-V1 model"""
    model = GoogLeNet(
        training=training,
        num_classes=num_classes,
        dropout=dropout,
        aux_dropout=aux_dropout,
        data_format=data_format,
    )
    if pretrained:
        url = "https://download.pytorch.org/models/googlenet-1378be20.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_inceptionNet_torch_weights_mapping,
        )
        model.v = w_clean
    return model
