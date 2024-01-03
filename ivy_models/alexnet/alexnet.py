import ivy
import ivy_models
from ivy_models.base import BaseSpec, BaseModel


class AlexNetSpec(BaseSpec):
    def __init__(self, num_classes=1000, dropout=0, data_format="NCHW"):
        super(AlexNetSpec, self).__init__(
            num_classes=num_classes, dropout=dropout, data_format=data_format
        )


class AlexNet(BaseModel):
    """An Ivy native implementation of AlexNet"""

    def __init__(
        self, num_classes=1000, dropout=0, data_format="NCHW", spec=None, v=None
    ):
        self.spec = (
            spec
            if spec and isinstance(spec, AlexNetSpec)
            else AlexNetSpec(
                num_classes=num_classes, dropout=dropout, data_format=data_format
            )
        )
        super(AlexNet, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.features = ivy.Sequential(
            ivy.Conv2D(3, 64, [11, 11], [4, 4], 2, data_format="NCHW"),
            ivy.ReLU(),
            ivy.MaxPool2D(3, 2, 0, data_format="NCHW"),
            ivy.Conv2D(64, 192, [5, 5], [1, 1], 2, data_format="NCHW"),
            ivy.ReLU(),
            ivy.MaxPool2D(3, 2, 0, data_format="NCHW"),
            ivy.Conv2D(192, 384, [3, 3], 1, 1, data_format="NCHW"),
            ivy.ReLU(),
            ivy.Conv2D(384, 256, [3, 3], 1, 1, data_format="NCHW"),
            ivy.ReLU(),
            ivy.Conv2D(256, 256, [3, 3], 1, 1, data_format="NCHW"),
            ivy.ReLU(),
            ivy.MaxPool2D(3, 2, 0, data_format="NCHW"),
        )
        self.avgpool = ivy.AdaptiveAvgPool2d((6, 6), data_format="NCHW")
        self.classifier = ivy.Sequential(
            ivy.Dropout(prob=self.spec.dropout),
            ivy.Linear(256 * 6 * 6, 4096),
            ivy.ReLU(),
            ivy.Dropout(prob=self.spec.dropout),
            ivy.Linear(4096, 4096),
            ivy.ReLU(),
            ivy.Linear(4096, self.spec.num_classes),
        )

    @classmethod
    def get_spec_class(self):
        return AlexNetSpec

    def _forward(self, x, data_format=None):
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NHWC":
            x = ivy.permute_dims(x, (0, 3, 1, 2))
        x = self.features(x)
        x = self.avgpool(x)
        x = ivy.reshape(x, (x.shape[0], -1))
        x = self.classifier(x)
        return x


def _alexnet_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key
    if "features" in old_key:
        if "bias" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "h -> 1 h 1 1"}
        elif "weight" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "b c h w-> h w c b"}
    return new_mapping


def alexnet(pretrained=True, num_classes=1000, dropout=0, data_format="NCHW"):
    """Ivy AlexNet model"""
    model = AlexNet(num_classes=num_classes, dropout=dropout, data_format=data_format)
    if pretrained:
        url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url, model, custom_mapping=_alexnet_torch_weights_mapping
        )
        model.v = w_clean
    return model
