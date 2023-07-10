from collections import OrderedDict
from typing import Tuple
import builtins

from ivy_models.helpers import load_torch_weights
import ivy


class DenseNetLayer(ivy.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float
    ) -> None:
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        self.bn_size = bn_size
        self.drop_rate = float(drop_rate)

        super().__init__()

    def _build(self, *args, **kwargs):
        self.norm1 = ivy.BatchNorm2D(self.num_input_features)
        self.relu1 = ivy.ReLU()
        self.conv1 = ivy.Conv2D(
            self.num_input_features,
            self.bn_size * self.growth_rate,
            [1, 1],
            1,
            0,
            with_bias=False,
            data_format="NCHW",
        )

        self.norm2 = ivy.BatchNorm2D(self.bn_size * self.growth_rate)
        self.relu2 = ivy.ReLU()
        self.conv2 = ivy.Conv2D(
            self.bn_size * self.growth_rate,
            self.growth_rate,
            [3, 3],
            1,
            1,
            with_bias=False,
            data_format="NCHW",
        )

    def bn_function(self, inputs):
        concated_features = ivy.concat(inputs, axis=1)
        bottleneck_output = self.conv1(
            self.relu1(self.norm1(concated_features))
        )  # noqa: T484
        return bottleneck_output

    # allowing it to take either a List[Tensor] or single Tensor
    def _forward(self, input):
        if isinstance(input, ivy.Array):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = ivy.dropout(
                new_features, prob=self.drop_rate, training=self.training
            )
        return new_features


class DenseNetBlock(ivy.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ) -> None:
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.bn_size = bn_size
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate

        super().__init__()

    def _build(self, *args, **kwargs):
        self.layers = OrderedDict()
        for i in range(self.num_layers):
            layer = DenseNetLayer(
                self.num_input_features + i * self.growth_rate,
                growth_rate=self.growth_rate,
                bn_size=self.bn_size,
                drop_rate=self.drop_rate,
            )
            self.layers["denselayer%d" % (i + 1)] = layer

    def _forward(self, init_features):
        features = [init_features]
        for name, layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
        return ivy.concat(features, axis=1)


class DenseNetTransition(ivy.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        self.num_input_features = num_input_features
        self.num_output_features = num_output_features

        super().__init__()

    def _build(self, *args, **kwargs):
        self.norm = ivy.BatchNorm2D(self.num_input_features)
        self.relu = ivy.ReLU()
        self.conv = ivy.Conv2D(
            self.num_input_features,
            self.num_output_features,
            [1, 1],
            1,
            0,
            with_bias=False,
            data_format="NCHW",
        )
        self.pool = ivy.AvgPool2D(2, 2, 0)


class DenseNet(ivy.Module):
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
        v=None,
    ) -> None:
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.num_init_features = num_init_features
        self.bn_size = bn_size
        self.drop_rate = drop_rate
        self.num_classes = num_classes

        super().__init__(v=v)

    def _build(self, *args, **kwargs):
        # First convolution
        layers = OrderedDict(
            [
                (
                    "conv0",
                    ivy.Conv2D(
                        3,
                        self.num_init_features,
                        [7, 7],
                        2,
                        3,
                        with_bias=False,
                        data_format="NCHW",
                    ),
                ),
                ("norm0", ivy.BatchNorm2D(self.num_init_features)),
                ("relu0", ivy.ReLU()),
                ("pool0", ivy.MaxPool2D(3, 2, 1, data_format="NCHW")),
            ]
        )

        # Each denseblock
        num_features = self.num_init_features
        for i, num_layers in enumerate(self.block_config):
            block = DenseNetBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=self.bn_size,
                growth_rate=self.growth_rate,
                drop_rate=self.drop_rate,
            )
            layers["denseblock%d" % (i + 1)] = block
            num_features = num_features + num_layers * self.growth_rate
            if i != len(self.block_config) - 1:
                trans = DenseNetTransition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                layers["transition%d" % (i + 1)] = trans
                num_features = num_features // 2

        # Final batch norm
        layers["norm5"] = ivy.BatchNorm2D(num_features)

        self.features = ivy.Sequential(layers)

        # Linear layer
        self.classifier = ivy.Linear(num_features, self.num_classes)

    def _forward(self, x):
        features = self.features(x)
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
            new_mapping = {"key_chain": new_key, "pattern": "b c h w-> h w c b"}
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
    ref_model = densenet(32, (6, 12, 24, 16), 64, v=v)
    if pretrained:
        url = "https://download.pytorch.org/models/densenet121-a639ec97.pth"
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_densenet_torch_weights_mapping,
        )
    return densenet(32, (6, 12, 24, 16), 64, v=w_clean)


def densenet161(v=None, pretrained=True):
    ref_model = densenet(48, (6, 12, 36, 24), 96, v=v)
    if pretrained:
        url = "https://download.pytorch.org/models/densenet161-17b70270.pth"
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_densenet_torch_weights_mapping,
        )
    return densenet(48, (6, 12, 36, 24), 96, v=w_clean)


def densenet169(v=None, pretrained=True):
    ref_model = densenet(32, (6, 12, 32, 32), 64, v=v)
    if pretrained:
        url = "https://download.pytorch.org/models/densenet169-b2777c0a.pth"
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_densenet_torch_weights_mapping,
        )
    return densenet(32, (6, 12, 32, 32), 64, v=w_clean)


def densenet201(v=None, pretrained=True):
    ref_model = densenet(32, (6, 12, 48, 32), 64, v=v)
    if pretrained:
        url = "https://download.pytorch.org/models/densenet201-c1103571.pth"
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=_densenet_torch_weights_mapping,
        )
    return densenet(32, (6, 12, 48, 32), 64, v=w_clean)
