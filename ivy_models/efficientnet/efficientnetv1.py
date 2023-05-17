# global
import ivy
import math


class CNNBlock(ivy.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
    ):
        """
        Helper module used in the MBConv and FusedMBConvBlock. Basic CNN
        block with batch norm and SiLU activation layer.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer.
        output_channels
            Number of output channels for the layer.
        kernel_size
            Size of the convolutional filter.
        stride
            The stride of the sliding window for each dimension of input.
        padding
            SAME" or "VALID" indicating the algorithm, or
            list indicating the per-dimension paddings.
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super(CNNBlock, self).__init__()

    def _build(self, *args, **kwargs) -> bool:
        self.conv = ivy.Sequential(
            ivy.Conv2D(
                self.input_channels,
                self.output_channels,
                [self.kernel_size, self.kernel_size],
                self.stride,
                self.padding,
                with_bias=False,
            ),
            ivy.BatchNorm2D(self.output_channels),
            ivy.SiLU(),
        )

    def _forward(self, x):
        return self.conv(x)


class SqueezeExcitation(ivy.Module):
    def __init__(self, input_channels, reduced_dim):
        """
        Helper module used in the MBConv and FusedMBConvBlock.

        Parameters
        ----------
        input_channels
            Number of input channels for the layer.
        reduced_dim
            Number of dimensionality reduction applied during the squeeze phase
        """
        self.conv1 = ivy.Conv2D(input_channels, reduced_dim, [1, 1], 1, "VALID")
        self.conv2 = ivy.Conv2D(reduced_dim, input_channels, [1, 1], 1, "VALID")
        self.silu = ivy.SiLU()
        super(SqueezeExcitation, self).__init__()

    def _forward(self, x):
        # N x H x W x C -> N x C x H x W
        x = ivy.reshape(x, shape=(x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
        x = ivy.adaptive_avg_pool2d(x, 1)  # C x H x W -> C x 1 x 1
        x = ivy.reshape(x, shape=(x.shape[0], x.shape[2], x.shape[3], x.shape[1]))
        x = self.conv1(x)
        x = self.silu(x)
        x = self.conv2(x)
        return self.silu(x)


class MBConvBlock(ivy.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        expand_ratio,
        padding="VALID",
        reduction_ratio=4,
        survival_prob=0.8,
    ):
        """
        Instantiates the Mobile Inverted Residual Bottleneck Block

        Parameters
        ----------
        input_channels
            Number of input channels for the layer.
        output_channels
            Number of output channels for the layer.
        kernel_size
            Size of the convolutional filter.
        stride
            The stride of the sliding window for each dimension of input.
        expand_ratio
            Degree of input channel expansion.
        padding
            SAME" or "VALID" indicating the algorithm, or
            list indicating the per-dimension paddings.
        reduction_ratio
            Dimensionality reduction in squeeze excitation.
        survival_prob
            Hyperparameter for stochastic depth.
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.use_residual = input_channels == output_channels and stride == 1
        self.hidden_dim = input_channels * expand_ratio
        self.expand = input_channels != self.hidden_dim
        self.reduced_dim = int(self.input_channels / reduction_ratio)
        self.reduction_ratio = reduction_ratio
        self.survival_prob = survival_prob
        super(MBConvBlock, self).__init__()

    def _build(self, *args, **kwrgs):
        if self.expand:
            self.expand_conv = CNNBlock(
                self.input_channels,
                self.hidden_dim,
                kernel_size=1,
                stride=1,
                padding=self.padding,
            )

        self.conv = ivy.Sequential(
            ivy.DepthwiseConv2D(
                self.hidden_dim,
                [self.kernel_size, self.kernel_size],
                self.stride,
                self.padding,
                with_bias=False,
            ),
            ivy.BatchNorm2D(self.hidden_dim),
            ivy.SiLU(),
            SqueezeExcitation(self.hidden_dim, self.reduced_dim),
            ivy.Conv2D(
                self.hidden_dim,
                self.output_channels,
                [1, 1],
                1,
                self.padding,
                with_bias=False,
            ),
            ivy.BatchNorm2D(self.output_channels),
        )

    def stochastic_depth(self, x):
        binary_tensor = (
            ivy.random_uniform(
                shape=(x.shape[0], 1, 1, 1), low=0, high=1, device=x.device
            )
            < self.survival_prob
        )
        return ivy.divide(x, self.survival_prob) * binary_tensor

    def _forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs
        x = self.conv(x)
        if self.use_residual:
            return self.stochastic_depth(x) + inputs
        return x


class EfficientNetV1(ivy.Module):
    def __init__(
        self,
        base_model,  # expand_ratio, channels, repeats, stride, kernel_size
        phi_values,  # phi_value, resolution, drop_rate
        num_classes,
        device="cuda:0",
        v: ivy.Container = None,
    ):
        """
        Instantiates the EfficientNetV1 architecture using given scaling
        coefficients.

        Parameters
        ----------
        base_model
            Base model configuration. Should contain expand_ratio,
            channels, repeats, stride, kernel_size
        phi_values
            Variant specific configuration. Should contain phi, resolution,
            dropout_rate
        num_classes
            Number of classes to classify images into.
        device
            device on which to create the model's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cuda:0.
        v
            the variables for the model, as a container, constructed internally
            by default.
        """
        self.base_model = base_model
        alpha = 1.2
        beta = 1.1
        self.dropout_rate = phi_values["dropout_rate"]
        self.depth_factor = alpha ** phi_values["phi"]
        self.width_factor = beta ** phi_values["phi"]
        self.num_classes = num_classes
        self.last_channels = math.ceil(1280 * self.width_factor)
        self.se_reduction_ratio = 4
        super(EfficientNetV1, self).__init__(v=v)

    def _build(self, *args, **kwrgs):
        channels = int(32 * self.width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for args in self.base_model:
            out_channels = self.se_reduction_ratio * math.ceil(
                int(args["channels"] * self.width_factor) / self.se_reduction_ratio
            )
            layers_repeats = math.ceil(args["repeats"] * self.depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=args["expand_ratio"],
                        stride=args["stride"] if layer == 0 else 1,
                        kernel_size=args["kernel_size"],
                        padding="SAME",
                        reduction_ratio=self.se_reduction_ratio,
                    )
                )
                in_channels = out_channels
        features.append(
            CNNBlock(
                in_channels, self.last_channels, kernel_size=1, stride=1, padding="SAME"
            )
        )

        self.features = ivy.Sequential(*features)

        self.classifier = ivy.Sequential(
            ivy.Dropout(self.dropout_rate),
            ivy.Linear(self.last_channels, self.num_classes),
        )

    def _forward(self, x):
        x = self.features(x)
        # N x H x W x C -> N x C x H x W
        x = ivy.reshape(x, shape=(x.shape[0], x.shape[3], x.shape[1], x.shape[2]))
        x = ivy.adaptive_avg_pool2d(x, 1)
        x = ivy.reshape(x, shape=(x.shape[0], -1))
        return self.classifier(x)


if __name__ == "__main__":
    import json

    # ivy.set_tensorflow_backend()
    ivy.set_jax_backend()
    import jax

    jax.config.update("jax_enable_x64", True)

    with open("variant_configs.json") as json_file:
        configs = json.load(json_file)

    configs = configs["v1"]
    base_model = configs["base_args"]
    phi_values = configs["phi_values"]["b0"]

    model = EfficientNetV1(
        base_model,
        phi_values,
        1000,
    )
    # print(model.v)

    res = phi_values["resolution"]
    x = ivy.random_normal(shape=(16, res, res, 3))
    print(model(x).shape)
