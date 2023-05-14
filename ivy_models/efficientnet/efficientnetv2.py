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
        expand_ratio,
        reduction_ratio=0.25, # squeeze excitation
        survival_prob=0.8,  # for stochastic depth
    ):
        """Mobile Inverted Residual Bottleneck Block

        TODO: complete docstring after code all good
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = "SAME"

        self.use_residual = input_channels == output_channels and stride == 1
        self.hidden_dim = input_channels * expand_ratio
        self.expand = input_channels != self.hidden_dim
        self.reduced_dim = int(self.hidden_dim * reduction_ratio)
        self.reduction_ratio = reduction_ratio
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

        conv = [
                ivy.DepthwiseConv2D(
                    self.hidden_dim, 
                    [self.kernel_size, self.kernel_size], 
                    self.stride, 
                    self.padding, 
                    with_bias=False
                )
            ]
        # Squeeze and excite
        if 0 < self.reduction_ratio <= 1:
            conv.append(SqueezeExcitation(self.hidden_dim, self.reduced_dim))
        
        conv += [
                ivy.Conv2D(self.hidden_dim, self.output_channels, [1, 1], 1, 'VALID', with_bias=False),
                ivy.BatchNorm2D(self.output_channels),
            ]

        self.conv = ivy.Sequential(*conv)

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


class FusedMBConvBlock(ivy.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
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
        self.expand = True if expand_ratio > 1 else False
        self.survival_prob = survival_prob
        super(FusedMBConvBlock, self).__init__()

    def _build(self, *args, **kwrgs):

        if self.expand:
            self.expand_conv = CNNBlock(
                self.input_channels, 
                self.output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        self.conv = ivy.Sequential(            
            ivy.Conv2D(
                self.hidden_dim, 
                self.output_channels, 
                [self.kernel_size, self.kernel_size] if self.expand else [1, 1],
                self.stride if self.expand else 1, 
                'SAME', 
                with_bias=False
            ),
            ivy.BatchNorm2D(
                self.output_channels
            )
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
    def __init__(self, version, num_classes, dropout_rate=0.1):
        # self.base_model = [
        #     # expand_ratio, channels, repeats, stride, kernel_size
        #     [1, 16, 1, 1, 3],
        #     [6, 24, 2, 2, 3],
        #     [6, 40, 2, 2, 5],
        #     [6, 80, 3, 2, 3],
        #     [6, 112, 3, 1, 5],
        #     [6, 192, 4, 2, 5],
        #     [6, 320, 1, 1, 3],
        # ]
        # phi_values = {
        #     # tuple of: (phi_value, resolution, drop_rate)
        #     "b0": (0, 224, 0.2),  # alpha, beta, gamma, depth = alpha ** phi
        #     "b1": (0.5, 240, 0.2),
        #     "b2": (1, 260, 0.3),
        #     "b3": (2, 300, 0.3),
        #     "b4": (3, 380, 0.4),
        #     "b5": (4, 456, 0.4),
        #     "b6": (5, 528, 0.5),
        #     "b7": (6, 600, 0.5),
        # }
        # alpha = 1.2
        # beta = 1.1
        # phi, res, self.dropout_rate = phi_values[version]
        variant_parameters = {
            "efficientnetv2-s": [
                {
                    "kernel_size": 3,
                    "num_repeat": 2,
                    "input_filters": 24,
                    "output_filters": 24,
                    "expand_ratio": 1,
                    "se_ratio": 0.0,
                    "strides": 1,
                    "conv_type": 1,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 4,
                    "input_filters": 24,
                    "output_filters": 48,
                    "expand_ratio": 4,
                    "se_ratio": 0.0,
                    "strides": 2,
                    "conv_type": 1,
                },
                {
                    "conv_type": 1,
                    "expand_ratio": 4,
                    "input_filters": 48,
                    "kernel_size": 3,
                    "num_repeat": 4,
                    "output_filters": 64,
                    "se_ratio": 0,
                    "strides": 2,
                },
                {
                    "conv_type": 0,
                    "expand_ratio": 4,
                    "input_filters": 64,
                    "kernel_size": 3,
                    "num_repeat": 6,
                    "output_filters": 128,
                    "se_ratio": 0.25,
                    "strides": 2,
                },
                {
                    "conv_type": 0,
                    "expand_ratio": 6,
                    "input_filters": 128,
                    "kernel_size": 3,
                    "num_repeat": 9,
                    "output_filters": 160,
                    "se_ratio": 0.25,
                    "strides": 1,
                },
                {
                    "conv_type": 0,
                    "expand_ratio": 6,
                    "input_filters": 160,
                    "kernel_size": 3,
                    "num_repeat": 15,
                    "output_filters": 256,
                    "se_ratio": 0.25,
                    "strides": 2,
                },
            ],
            "efficientnetv2-m": [
                {
                    "kernel_size": 3,
                    "num_repeat": 3,
                    "input_filters": 24,
                    "output_filters": 24,
                    "expand_ratio": 1,
                    "se_ratio": 0,
                    "strides": 1,
                    "conv_type": 1,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 5,
                    "input_filters": 24,
                    "output_filters": 48,
                    "expand_ratio": 4,
                    "se_ratio": 0,
                    "strides": 2,
                    "conv_type": 1,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 5,
                    "input_filters": 48,
                    "output_filters": 80,
                    "expand_ratio": 4,
                    "se_ratio": 0,
                    "strides": 2,
                    "conv_type": 1,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 7,
                    "input_filters": 80,
                    "output_filters": 160,
                    "expand_ratio": 4,
                    "se_ratio": 0.25,
                    "strides": 2,
                    "conv_type": 0,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 14,
                    "input_filters": 160,
                    "output_filters": 176,
                    "expand_ratio": 6,
                    "se_ratio": 0.25,
                    "strides": 1,
                    "conv_type": 0,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 18,
                    "input_filters": 176,
                    "output_filters": 304,
                    "expand_ratio": 6,
                    "se_ratio": 0.25,
                    "strides": 2,
                    "conv_type": 0,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 5,
                    "input_filters": 304,
                    "output_filters": 512,
                    "expand_ratio": 6,
                    "se_ratio": 0.25,
                    "strides": 1,
                    "conv_type": 0,
                },
            ],
            "efficientnetv2-l": [
                {
                    "kernel_size": 3,
                    "num_repeat": 4,
                    "input_filters": 32,
                    "output_filters": 32,
                    "expand_ratio": 1,
                    "se_ratio": 0,
                    "strides": 1,
                    "conv_type": 1,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 7,
                    "input_filters": 32,
                    "output_filters": 64,
                    "expand_ratio": 4,
                    "se_ratio": 0,
                    "strides": 2,
                    "conv_type": 1,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 7,
                    "input_filters": 64,
                    "output_filters": 96,
                    "expand_ratio": 4,
                    "se_ratio": 0,
                    "strides": 2,
                    "conv_type": 1,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 10,
                    "input_filters": 96,
                    "output_filters": 192,
                    "expand_ratio": 4,
                    "se_ratio": 0.25,
                    "strides": 2,
                    "conv_type": 0,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 19,
                    "input_filters": 192,
                    "output_filters": 224,
                    "expand_ratio": 6,
                    "se_ratio": 0.25,
                    "strides": 1,
                    "conv_type": 0,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 25,
                    "input_filters": 224,
                    "output_filters": 384,
                    "expand_ratio": 6,
                    "se_ratio": 0.25,
                    "strides": 2,
                    "conv_type": 0,
                },
                {
                    "kernel_size": 3,
                    "num_repeat": 7,
                    "input_filters": 384,
                    "output_filters": 640,
                    "expand_ratio": 6,
                    "se_ratio": 0.25,
                    "strides": 1,
                    "conv_type": 0,
                },
            ],
        }
        self.model_blocks = variant_parameters[version]
        self.depth_factor = 1.
        self.width_factor = 1.
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.last_channels = math.ceil(1280 * self.width_factor)
        # self.se_reduction_ratio = 4
        super(EfficientNet, self).__init__()
    '''
            {
                "kernel_size": 3,
                "num_repeat": 2,
                "input_filters": 24,
                "output_filters": 24,
                "expand_ratio": 1,
                "se_ratio": 0.0,
                "strides": 1,
                "conv_type": 1,
            },
    '''

    def _build(self, *args, **kwrgs):
        channels = int(self.model_blocks[0]['input_filters'] * self.width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        

        for i, args in self.model_blocks:
            layers_repeats = math.ceil(args.num_repeat * self.depth_factor)

            in_channels = channels
            # se_reduction_ratio = args.se_ratio
            # not too sure about out_channels, check the math later
            out_channels = self.se_reduction_ratio * \
                math.ceil(int(channels * self.width_factor) / self.se_reduction_ratio)

            for layer_stage in range(layers_repeats):
                if layer_stage <= 2:
                    features.append(
                        FusedMBConvBlock(
                            in_channels,
                            out_channels,
                            expand_ratio=args.expand_ratio,
                            stride=args.strides,
                            kernel_size=args.kernel_size,
                        )
                    )
                else:
                    features.append(
                        MBConvBlock(
                            in_channels,
                            out_channels,
                            expand_ratio=args.expand_ratio,
                            stride=args.strides,
                            kernel_size=args.kernel_size,
                            reduction_ratio=args.se_ratio,
                        )
                    )
                in_channels = out_channels
        features.append(
            CNNBlock(in_channels, self.last_channels, kernel_size=1, stride=1, padding="SAME")
        )

        self.features = ivy.Sequential(*features)

        self.classifier = ivy.Sequential(
            ivy.Dropout(self.dropout_rate),
            ivy.Linear(self.last_channels, self.num_classes),
        )


    def _forward(self, x):
        x = ivy.adaptive_avg_pool2d(self.features(x), 1)
        return self.classifier(ivy.reshape(x, shape=(x.shape[0], -1)))

