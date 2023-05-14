# global
import ivy
import math


class CNNBlock(ivy.Module):
    def __init__(self, 
        in_channels, 
        out_channels, 
        kernel_size, 
        stride, 
        padding,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        super(CNNBlock, self).__init__()
    
    def _build(self, *args, **kwargs) -> bool:
        self.conv = ivy.Conv2D(
                    self.in_channels, 
                    self.out_channels, 
                    [self.kernel_size, self.kernel_size], 
                    self.stride, 
                    self.padding,
                    with_bias=False
                )
        self.batch_norm = ivy.BatchNorm2D(self.out_channels)

    def _forward(self, x):
        x = self.batch_norm(self.conv(x))
        # SiLU activation, refactor after it's added in Ivy
        # https://github.com/unifyai/ivy/pull/15385
        return x * ivy.sigmoid(x)
        

class SqueezeExcitation(ivy.Module):
    def __init__(self, input_channels, reduced_dim):
        self.conv1 = ivy.Conv2D(input_channels, reduced_dim, [1, 1], 1, 'VALID')
        self.conv2 = ivy.Conv2D(reduced_dim, input_channels, [1, 1], 1, 'VALID')
        super(SqueezeExcitation, self).__init__()

    def _forward(self, x):
        se = ivy.adaptive_avg_pool2d(x, 1) # C x H x W -> C x 1 x 1
        se = self.conv1(se)
        se = se * ivy.sigmoid(se) # SiLU
        se = self.conv2(se)
        se = ivy.sigmoid(se)
        return x * se


class MBConvBlock(ivy.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction_ratio=4, # squeeze excitation
        survival_prob=0.8,  # for stochastic depth
    ):
        """Mobile Inverted Residual Bottleneck Block

        TODO: complete docstring after code all good
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.use_residual = input_channels == output_channels and stride == 1
        self.hidden_dim = input_channels * expand_ratio
        self.expand = input_channels != self.hidden_dim
        self.reduced_dim = int(self.hidden_dim / reduction_ratio)
        self.survival_prob = survival_prob
        super(MBConvBlock, self).__init__()

    def _build(self, *args, **kwrgs):

        if self.expand:
            self.expand_conv = CNNBlock(
            self.input_channels, 
            self.output_channels,
            kernel_size=1,
            stride=1,
            padding=1,
            )

        self.conv = ivy.Sequential(
            ivy.DepthwiseConv2D(self.hidden_dim, [self.kernel_size, self.kernel_size], self.stride, self.padding, with_bias=False),
            SqueezeExcitation(self.hidden_dim, self.reduced_dim),
            ivy.Conv2D(self.hidden_dim, self.output_channels, [1, 1], 1, 'VALID', with_bias=False),
            ivy.BatchNorm2D(self.output_channels),
        )

    def stochastic_depth(self, x):
        binary_tensor = (
            ivy.random_uniform(shape=(x.shape[0], 1, 1, 1), low=0, high=1, device=x.device) < self.survival_prob
        )
        return ivy.divide(x, self.survival_prob) * binary_tensor


    def _forward(self, inputs):
        x = self.expand_conv(inputs) if self.expand else inputs

        if self.use_residual:
            return self.stochastic_depth(self.conv(x)) + inputs
        else:
            return self.conv(x)


class EfficientNet(ivy.Module):
    def __init__(self, version, num_classes):
        self.base_model = [
            # expand_ratio, channels, repeats, stride, kernel_size
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        phi_values = {
            # tuple of: (phi_value, resolution, drop_rate)
            "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
            "b1": (0.5, 240, 0.2),
            "b2": (1, 260, 0.3),
            "b3": (2, 300, 0.3),
            "b4": (3, 380, 0.4),
            "b5": (4, 456, 0.4),
            "b6": (5, 528, 0.5),
            "b7": (6, 600, 0.5),
        }
        alpha = 1.2
        beta = 1.1
        phi, res, self.dropout_rate = phi_values[version]
        self.depth_factor = alpha**phi
        self.width_factor = beta**phi
        self.num_classes = num_classes
        self.last_channels = math.ceil(1280 * self.width_factor)
        self.se_reduction_ratio = 4
        super(EfficientNet, self).__init__()


    def _build(self, *args, **kwrgs):
        channels = int(32 * self.width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for expand_ratio, channels, repeats, stride, kernel_size in self.base_model:
            out_channels = self.se_reduction_ratio * \
                math.ceil(int(channels * self.width_factor) / self.se_reduction_ratio)
            layers_repeats = math.ceil(repeats * self.depth_factor)

            for layer in range(layers_repeats):
                features.append(
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        expand_ratio=expand_ratio,
                        stride=stride if layer == 0 else 1,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                        reduction_ratio=self.se_reduction_ratio,
                    )
                )
                in_channels = out_channels
        features.append(
            CNNBlock(in_channels, self.last_channels, kernel_size=1, stride=1, padding=0)
        )

        self.features = ivy.Sequential(*features)

        self.classifier = ivy.Sequential(
            ivy.Dropout(self.dropout_rate),
            ivy.Linear(self.last_channels, self.num_classes),
        )


    def _forward(self, x):
        x = ivy.adaptive_avg_pool2d(self.features(x), 1)
        return self.classifier(ivy.reshape(x, shape=(x.shape[0], -1)))

