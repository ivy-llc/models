from collections import OrderedDict
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
        )

    def bn_function(self, inputs):
        concated_features = ivy.concat(inputs, axis=3)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
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
        for layer in list(self.layers.values()):
            new_features = layer(features)
            features.append(new_features)
        return ivy.concat(features, axis=3)


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
        )
        self.pool = ivy.AvgPool2D(2, 2, 0)
