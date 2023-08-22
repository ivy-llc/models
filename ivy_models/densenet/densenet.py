from collections import OrderedDict
from typing import Tuple
import builtins

from ivy_models.helpers import load_torch_weights
from ivy_models.densenet.denselayers import DenseNetBlock, DenseNetTransition, ivy
from ivy_models.base import BaseSpec, BaseModel


class DenseNetLayerSpec(BaseSpec):
    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
    ) -> None:
        super(DenseNetLayerSpec, self).__init__(
            growth_rate=growth_rate,
            block_config=block_config,
            num_init_features=num_init_features,
            bn_size=bn_size,
            drop_rate=drop_rate,
            num_classes=num_classes,
        )


class DenseNet(BaseModel):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks"
    <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
    ----
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn
          in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
          but slower. Default: *False*.
          See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 1000,
        spec=None,
        v=None,
    ) -> None:
        self.spec = (
            spec
            if spec and isinstance(spec, DenseNetLayerSpec)
            else DenseNetLayerSpec(
                growth_rate=growth_rate,
                block_config=block_config,
                num_init_features=num_init_features,
                bn_size=bn_size,
                drop_rate=drop_rate,
                num_classes=num_classes,
            )
        )
        super().__init__(v=v)

    def _build(self, *args, **kwargs):
        # First convolution
        layers = OrderedDict(
            [
                (
                    "conv0",
                    ivy.Conv2D(
                        3,
                        self.spec.num_init_features,
                        [7, 7],
                        2,
                        3,
                        with_bias=False,
                    ),
                ),
                (
                    "norm0",
                    ivy.BatchNorm2D(self.spec.num_init_features),
                ),
                ("relu0", ivy.ReLU()),
                ("pool0", ivy.MaxPool2D(3, 2, 1)),
            ]
        )

        # Each denseblock
        num_features = self.spec.num_init_features
        for i, num_layers in enumerate(self.spec.block_config):
            block = DenseNetBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=self.spec.bn_size,
                growth_rate=self.spec.growth_rate,
                drop_rate=self.spec.drop_rate,
            )
            layers["denseblock%d" % (i + 1)] = block
            num_features = num_features + num_layers * self.spec.growth_rate
            if i != len(self.spec.block_config) - 1:
                trans = DenseNetTransition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                layers["transition%d" % (i + 1)] = trans
                num_features = num_features // 2

        # Final batch norm
        layers["norm5"] = ivy.BatchNorm2D(num_features)

        self.features = layers  # ivy.Sequential(layers)

        # Linear layer
        self.classifier = ivy.Linear(num_features, self.spec.num_classes)

    @classmethod
    def get_spec_class(self):
        return DenseNetLayerSpec

    def _forward(self, x):
        templist = list(self.features.values())
        layers = ivy.Sequential(*templist)
        features = layers(x)
        out = ivy.relu(features)
        out = ivy.adaptive_avg_pool2d(out, (1, 1))
        out = ivy.flatten(out, axis=1)
        out = self.classifier(out)
        return out


def _densenet_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key

    if "features" in old_key:
        W_KEY = ["conv0/weight", "conv/1/weight", "conv/2/weight", "conv/weight"]
        if builtins.any([kc in old_key for kc in W_KEY]):
            new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    return new_mapping


def densenet(
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    v=None,
):
    model = DenseNet(growth_rate, block_config, num_init_features, v=v)
    return model


def densenet121(v=None, pretrained=True):
    model = densenet(32, (6, 12, 24, 16), 64, v=v)
    if pretrained:
        url = "https://download.pytorch.org/models/densenet121-a639ec97.pth"
        w_clean = load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_densenet_torch_weights_mapping,
        )
        model.v = w_clean

    return model


def densenet161(v=None, pretrained=True):
    model = densenet(48, (6, 12, 36, 24), 96, v=v)
    if pretrained:
        url = "https://download.pytorch.org/models/densenet161-17b70270.pth"
        w_clean = load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_densenet_torch_weights_mapping,
        )
        model.v = w_clean
    return model


def densenet169(v=None, pretrained=True):
    model = densenet(32, (6, 12, 32, 32), 64, v=v)
    if pretrained:
        url = "https://download.pytorch.org/models/densenet169-b2777c0a.pth"
        w_clean = load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_densenet_torch_weights_mapping,
        )
        model.v = w_clean
    return model


def densenet201(v=None, pretrained=True):
    model = densenet(32, (6, 12, 48, 32), 64, v=v)
    if pretrained:
        url = "https://download.pytorch.org/models/densenet201-c1103571.pth"
        w_clean = load_torch_weights(
            url,
            model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_densenet_torch_weights_mapping,
        )
        model.v = w_clean
    return model


if __name__ == "__main__":
    ivy.set_torch_backend()
    model = densenet(32, (6, 12, 24, 16), 64)
    print(model.v)
