from ivy_models.helpers import load_torch_weights
import ivy
import builtins

class Fire(ivy.Module):
    def __init__(self, inplanes: int, squeeze_planes: int, expand1x1_planes: int, expand3x3_planes: int) -> None:
        self.inplanes = inplanes
        self.squeeze = ivy.Conv2D(inplanes, squeeze_planes, [1, 1], 1, 0)
        self.squeeze_activation = ivy.ReLU()
        self.expand1x1 = ivy.Conv2D(squeeze_planes, expand1x1_planes, [1, 1], 1, 0)
        self.expand1x1_activation = ivy.ReLU()
        self.expand3x3 = ivy.Conv2D(squeeze_planes, expand3x3_planes, [3, 3], 1, 1)
        self.expand3x3_activation = ivy.ReLU()
        super().__init__()

    def _forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return ivy.concat(
            [self.expand1x1_activation(self.expand1x1(x)), self.expand3x3_activation(self.expand3x3(x))], axis=3
        )
    

class SqueezeNet(ivy.Module):
    def __init__(self, version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5, v=None) -> None:
        self.num_classes = num_classes
        if version == "1_0":
            self.features = ivy.Sequential(
                ivy.Conv2D(3, 96, [7, 7], 2, 0),
                ivy.ReLU(),
                ivy.MaxPool2D(3, 2, 0),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                ivy.MaxPool2D(3, 2, 0),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                ivy.MaxPool2D(3, 2, 0),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = ivy.Sequential(
                ivy.Conv2D(3, 64, [3, 3], 2, 0),
                ivy.ReLU(),
                ivy.MaxPool2D(3, 2, 0),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                ivy.MaxPool2D(3, 2, 0),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                ivy.MaxPool2D(3, 2, 0),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        # Final convolution is initialized differently from the rest
        final_conv = ivy.Conv2D(512, self.num_classes, [1, 1], 1, 0)
        self.classifier = ivy.Sequential(
            ivy.Dropout(prob=dropout), final_conv, ivy.ReLU(), ivy.AdaptiveAvgPool2d((1, 1))
        )
        super().__init__(v=v)

    def _forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return ivy.flatten(x, 1)
    

def _squeezenet_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key
    if "weight" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "b c h w -> w h c b"}
    elif "bias" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "h -> 1 1 1 h"}

    return new_mapping


def squeezeNet(version: str = "1_0", num_classes: int = 1000, dropout: float = 0.5, v=None, pretrained=True):
    if not pretrained:
        return SqueezeNet(version, num_classes, dropout, v=v)

    reference_model = SqueezeNet(version, num_classes, dropout, v=v)
    if version == "1_0":
        url="https://download.pytorch.org/models/squeezenet1_0-b66bff10.pth"
    elif version == "1_1":
        url="https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth"
    else:
        raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")
    w_clean = load_torch_weights(
        url,
        reference_model,
        raw_keys_to_prune=["num_batches_tracked"],
        custom_mapping=_squeezenet_torch_weights_mapping,
    )
    return SqueezeNet(version, num_classes, dropout, v=w_clean)
