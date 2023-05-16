import ivy
import math


class CNNBlock(ivy.Module):
    def __init__(self, 
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
                        with_bias=False
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
        """
        self.conv1 = ivy.Conv2D(input_channels, reduced_dim, [1, 1], 1, 'VALID')
        self.conv2 = ivy.Conv2D(reduced_dim, input_channels, [1, 1], 1, 'VALID')
        self.silu = ivy.SiLU()
        super(SqueezeExcitation, self).__init__()

    def _forward(self, x):
        # N x H x W x C -> N x C x H x W
        x = ivy.reshape( 
                x, 
                shape=(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
            ) 
        x = ivy.adaptive_avg_pool2d(x, 1)  # C x H x W -> C x 1 x 1
        x = ivy.reshape( 
                x, 
                shape=(x.shape[0], x.shape[2], x.shape[3], x.shape[1])
            ) 
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
        reduction_ratio=0.25, 
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
        self.reduced_dim = int(self.hidden_dim * reduction_ratio)
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
        print('mbconv')
        print('1', inputs.shape)
        x = self.expand_conv(inputs) if self.expand else inputs
        print('2', x.shape)
        x = self.conv(x)
        print('3', x.shape)
        if self.use_residual:
            return self.stochastic_depth(x) + inputs
        return x


class FusedMBConvBlock(ivy.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        expand_ratio,
        padding='VALID',
        survival_prob=0.8, 
    ):
        """
        Instantiates the Fused Mobile Inverted Residual Bottleneck Block 

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
        self.survival_prob = survival_prob
        super(FusedMBConvBlock, self).__init__()

    def _build(self, *args, **kwrgs):

        if self.expand:
            self.expand_conv = CNNBlock(
                self.input_channels, 
                self.hidden_dim,
                kernel_size=3,
                stride=1,
                padding=self.padding,
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
        print('1', inputs.shape)
        x = self.expand_conv(inputs) if self.expand else inputs
        print('2', x.shape)
        print(self.input_channels, self.hidden_dim, self.output_channels, self.kernel_size, self.stride)
        x = self.conv(x)
        print('3', x.shape)
        if self.use_residual:
            return self.stochastic_depth(x) + inputs
        return x


class EfficientNetV2(ivy.Module):
    def __init__(
            self, 
            config,
            num_classes, 
            depth_divisor=8,
            min_depth=8,
            dropout_rate=0.1,
            device='cuda:0',
            v: ivy.Container = None,
        ):
        """
        Instantiates the EfficientNetV2 architecture using given scaling
        coefficients.

        Parameters
        ----------
        config
            Variant specific configuration. Should contain scaling coefficients,
            individual block's kernel_size, num_repeat, input_filters, output_filters, 
            expand_ratio, se_ratio, strides, conv_type
        num_classes
            Number of classes to classify images into.
        depth_divisor
            A unit of network width
        min_depth
            Minimum number of filters
        dropout_rate
            Dropout rate before final classifier layer
        device
            device on which to create the model's variables 'cuda:0', 'cuda:1', 'cpu'
            etc. Default is cuda:0.
        v
            the variables for the model, as a container, constructed internally
            by default.
        """
        self.model_blocks = config["blocks"]
        self.depth_factor = config["phi_values"]["depth_coefficient"]
        self.width_factor = config["phi_values"]["width_coefficient"]
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.depth_divisor = depth_divisor
        self.min_depth = min_depth
        self.last_channels = math.ceil(1280 * self.width_factor)
        super(EfficientNetV2, self).__init__(v=v)

    @staticmethod
    def round_filters(filters, width_coefficient, min_depth, depth_divisor):
        """Round number of filters based on depth multiplier."""
        filters *= width_coefficient
        minimum_depth = min_depth or depth_divisor
        new_filters = max(
            minimum_depth,
            int(filters + depth_divisor / 2) // depth_divisor * depth_divisor,
        )
        return int(new_filters)


    def _build(self, *args, **kwrgs):
        channels = int(self.model_blocks[0]['input_filters'] * self.width_factor)
        features = [CNNBlock(3, channels, 3, stride=2, padding=1)]
        
        for args in self.model_blocks:
            layers_repeats = math.ceil(args['num_repeat'] * self.depth_factor)

            in_channels = self.round_filters(
                args['input_filters'],
                self.width_factor,
                self.min_depth,
                self.depth_divisor
            )
            out_channels = self.round_filters(
                args['output_filters'],
                self.width_factor,
                self.min_depth,
                self.depth_divisor
            )

            for layer_stage in range(layers_repeats):
                if layer_stage <= 2:
                    features.append(
                        FusedMBConvBlock(
                            in_channels,
                            out_channels,
                            expand_ratio=args['expand_ratio'],
                            stride=args['strides'],
                            kernel_size=args['kernel_size'],
                        )
                    )
                else:
                    features.append(
                        MBConvBlock(
                            in_channels,
                            out_channels,
                            expand_ratio=args['expand_ratio'],
                            stride=args['strides'],
                            kernel_size=args['kernel_size'],
                            reduction_ratio=args['se_ratio'],
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
        x = self.features(x)
        # N x H x W x C -> N x C x H x W
        x = ivy.reshape( 
                x, 
                shape=(x.shape[0], x.shape[3], x.shape[1], x.shape[2])
            ) 
        x = ivy.adaptive_avg_pool2d(x, 1)
        x = ivy.reshape(x, shape=(x.shape[0], -1))
        return self.classifier(x)


if __name__ == "__main__":
    import json
    # ivy.set_tensorflow_backend()
    ivy.set_jax_backend()
    import jax
    jax.config.update('jax_enable_x64', True)
    
    with open("variant_configs.json") as json_file:
        configs = json.load(json_file)

    configs = configs["v2"]["efficientnetv2-b0"]

    model = EfficientNetV2(
            configs, 
            10
        )
    # print(model.v)

    res = configs["phi_values"]['resolution']
    x = ivy.random_normal(shape=(16, res, res, 3))
    print(type(x), x.device, x.shape)
    print(model(x).shape)
