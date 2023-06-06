import ivy


def vgg_conv_block(inp_channels, out_channels, with_bn=False):
    """Vgg Conv Layer"""
    conv_block = [
        ivy.Conv2D(inp_channels, out_channels, [3, 3], 1, 1),
        ivy.ReLU(),
    ]
    if with_bn:
        bn_layer = ivy.BatchNorm2D(out_channels)
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


class VGG(ivy.Module):
    def __init__(self, repeats, with_bn=False, num_classes=1000, v=None):
        self.training = False
        layers = []
        filters = [3, 64, 128, 256, 512, 512]
        for idx, rep in enumerate(repeats):
            layers += vgg_block(
                inp_channel=filters[idx],
                out_channel=filters[idx + 1],
                repeat=rep,
                with_bn=with_bn,
            )
        self.features = ivy.Sequential(*layers)
        self.classifier = ivy.Sequential(
            ivy.Linear(7 * 7 * 512, 4096),
            ivy.ReLU(),
            ivy.Dropout(prob=0.5, training=self.training),
            ivy.Linear(4096, 4096),
            ivy.ReLU(),
            ivy.Dropout(prob=0.5, training=self.training),
            ivy.Linear(4096, num_classes),
        )
        super(VGG, self).__init__(v=v)

    def _forward(self, inputs):
        inputs = self.features(inputs)
        inputs = ivy.permute_dims(inputs, (0, 3, 2, 1)).reshape((inputs.shape[0], -1))
        return self.classifier(inputs)


def vgg11(v=None):
    """VGG11 model"""
    return VGG([1, 1, 2, 2, 2], False, v=v)


def vgg11_bn(v=None):
    """VGG11 model with BatchNorm2D"""
    return VGG([1, 1, 2, 2, 2], True, v=v)


def vgg13(v=None):
    """VGG13 model"""
    return VGG([2, 2, 2, 2, 2], False, v=v)


def vgg13_bn(v=None):
    """VGG13 model with BatchNorm2D"""
    return VGG([2, 2, 2, 2, 2], True, v=v)


def vgg16(v=None):
    """VGG16 model"""
    return VGG([2, 2, 3, 3, 3], False, v=v)


def vgg16_bn(v=None):
    """VGG16 model with BatchNorm2D"""
    return VGG([2, 2, 3, 3, 3], True, v=v)


def vgg19(v=None):
    """VGG19 model"""
    return VGG([2, 2, 4, 4, 4], False, v=v)


def vgg19_bn(v=None):
    """VGG19 model with BatchNorm2D"""
    return VGG([2, 2, 4, 4, 4], True, v=v)
