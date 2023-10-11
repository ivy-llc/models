import ivy
import ivy_models
from ivy_models.base import BaseSpec, BaseModel


def vgg_conv_block(inp_channels, out_channels, with_bn=False):
    """Vgg Conv Layer"""
    conv_block = [
        ivy.Conv2D(inp_channels, out_channels, [3, 3], 1, 1),
        ivy.ReLU(),
    ]
    if with_bn:
        bn_layer = ivy.BatchNorm2D(out_channels, training=False)
        conv_block.insert(1, bn_layer)

    return conv_block


def vgg_block(inp_channel, out_channel, repeat, with_bn=False):
    """Vgg Block"""
    layers = []
    layers += vgg_conv_block(inp_channel, out_channel, with_bn)
    for idx in range(1, repeat):
        layers += vgg_conv_block(out_channel, out_channel, with_bn)
    layers.append(ivy.MaxPool2D(2, 2, 0))
    return layers


class VGGSpec(BaseSpec):
    def __init__(self, repeats, with_bn=False, num_classes=1000, data_format="NHWC"):
        super(VGGSpec, self).__init__(
            repeats=repeats,
            with_bn=with_bn,
            num_classes=num_classes,
            data_format=data_format,
        )


class VGG(BaseModel):
    def __init__(
        self,
        repeats,
        with_bn=False,
        num_classes=1000,
        spec=None,
        data_format="NHWC",
        v=None,
    ):
        self.spec = (
            spec
            if spec and isinstance(spec, VGGSpec)
            else VGGSpec(
                repeats=repeats,
                with_bn=with_bn,
                num_classes=num_classes,
                data_format=data_format,
            )
        )
        super(VGG, self).__init__(v=v)

    def _build(self, *args, **kwargs):
        self.training = False
        layers = []
        filters = [3, 64, 128, 256, 512, 512]
        for idx, rep in enumerate(self.spec.repeats):
            layers += vgg_block(
                inp_channel=filters[idx],
                out_channel=filters[idx + 1],
                repeat=rep,
                with_bn=self.spec.with_bn,
            )
        self.features = ivy.Sequential(*layers)
        self.classifier = ivy.Sequential(
            ivy.Linear(7 * 7 * 512, 4096),
            ivy.ReLU(),
            ivy.Dropout(prob=0.5, training=self.training),
            ivy.Linear(4096, 4096),
            ivy.ReLU(),
            ivy.Dropout(prob=0.5, training=self.training),
            ivy.Linear(4096, self.spec.num_classes),
        )

    @classmethod
    def get_spec_class(self):
        return VGGSpec

    def _forward(self, inputs, data_format):
        data_format = data_format if data_format else self.spec.data_format
        if data_format == "NCHW":
            inputs = ivy.permute_dims(inputs, (0, 2, 3, 1))
        inputs = self.features(inputs)
        inputs = ivy.permute_dims(inputs, (0, 3, 2, 1)).reshape((inputs.shape[0], -1))
        return self.classifier(inputs)


def _vgg_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key
    if "features" in old_key:
        if "bias" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "h -> 1 1 1 h"}
        elif "weight" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "b c h w-> w h c b"}
    return new_mapping


def vgg11(pretrained=True, data_format="NHWC"):
    """VGG11 model"""
    model = VGG([1, 1, 2, 2, 2], False)
    if pretrained:
        url = "https://download.pytorch.org/models/vgg11-8a719046.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url, model, custom_mapping=_vgg_torch_weights_mapping
        )
        model.v.cont_set_at_key_chains(w_clean, inplace=True)
    return model


def vgg11_bn(pretrained=True, data_format="NHWC"):
    """VGG11 model with BatchNorm2D"""
    model = VGG([1, 1, 2, 2, 2], True)
    if pretrained:
        url = "https://download.pytorch.org/models/vgg11_bn-6002323d.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url, model, custom_mapping=_vgg_torch_weights_mapping
        )
        model.v.cont_set_at_key_chains(w_clean, inplace=True)
    return model


def vgg13(pretrained=True, data_format="NHWC"):
    """VGG13 model"""
    model = VGG([2, 2, 2, 2, 2], False)
    if pretrained:
        url = "https://download.pytorch.org/models/vgg13-19584684.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url, model, custom_mapping=_vgg_torch_weights_mapping
        )
        model.v.cont_set_at_key_chains(w_clean, inplace=True)
    return model


def vgg13_bn(pretrained=True, data_format="NHWC"):
    """VGG13 model with BatchNorm2D"""
    model = VGG([2, 2, 2, 2, 2], True)
    if pretrained:
        url = "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url, model, custom_mapping=_vgg_torch_weights_mapping
        )
        model.v.cont_set_at_key_chains(w_clean, inplace=True)
    return model


def vgg16(pretrained=True, data_format="NHWC"):
    """VGG16 model"""
    model = VGG([2, 2, 3, 3, 3], False)
    if pretrained:
        url = "https://download.pytorch.org/models/vgg16-397923af.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url, model, custom_mapping=_vgg_torch_weights_mapping
        )
        model.v.cont_set_at_key_chains(w_clean, inplace=True)
    return model


def vgg16_bn(pretrained=True, data_format="NHWC"):
    """VGG16 model with BatchNorm2D"""
    model = VGG([2, 2, 3, 3, 3], True)
    if pretrained:
        url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url, model, custom_mapping=_vgg_torch_weights_mapping
        )
        model.v.cont_set_at_key_chains(w_clean, inplace=True)
    return model


def vgg19(pretrained=True, data_format="NHWC"):
    """VGG19 model"""
    model = VGG([2, 2, 4, 4, 4], False)
    if pretrained:
        url = "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url, model, custom_mapping=_vgg_torch_weights_mapping
        )
        model.v.cont_set_at_key_chains(w_clean, inplace=True)
    return model


def vgg19_bn(pretrained=True, data_format="NHWC"):
    """VGG19 model with BatchNorm2D"""
    model = VGG([2, 2, 4, 4, 4], True)
    if pretrained:
        url = "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth"
        w_clean = ivy_models.helpers.load_torch_weights(
            url, model, custom_mapping=_vgg_torch_weights_mapping
        )
        model.v.cont_set_at_key_chains(w_clean, inplace=True)
    return model


if __name__ == "__main__":
    ivy.set_torch_backend()
    model = vgg11()
