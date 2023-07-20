import ivy
import ivy_models
from .layers import ConvNeXtBlock, ConvNeXtV2Block, ConvNeXtLayerNorm


class ConvNeXt(ivy.Module):
    def __init__(
        self,
        version=1,
        in_channels=3,
        num_classes=1000,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        device=None,
        training=False,
        v=None,
    ):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.depths = depths
        self.dims = dims
        self.drop_path_rate = drop_path_rate
        self.layer_scale_init_value = layer_scale_init_value
        self.head_init_scale = head_init_scale
        assert version == 1 or version == 2
        self.version = version
        super(ConvNeXt, self).__init__(device=device, v=v)

    def _build(self, *args, **kwargs):
        self.downsample_layers = []
        stem = ivy.Sequential(
            ivy.Conv2D(
                self.in_channels, self.dims[0], [4, 4], (4, 4), 0, data_format="NCHW"
            ),
            ConvNeXtLayerNorm(self.dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = ivy.Sequential(
                ConvNeXtLayerNorm(self.dims[i], eps=1e-6, data_format="channels_first"),
                ivy.Conv2D(
                    self.dims[i],
                    self.dims[i + 1],
                    [2, 2],
                    (2, 2),
                    0,
                    data_format="NCHW",
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = []
        dp_rates = [x for x in ivy.linspace(0, self.drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(4):
            stage = ivy.Sequential(
                *[
                    ConvNeXtBlock(
                        dim=self.dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=self.layer_scale_init_value,
                    )
                    if self.version == 1
                    else ConvNeXtV2Block(
                        dim=self.dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=self.layer_scale_init_value,
                    )
                    for j in range(self.depths[i])
                ]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        self.norm = ivy.LayerNorm(self.dims[-1], eps=1e-6)
        self.head = ivy.Linear(self.dims[-1], self.num_classes)

    def _forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(ivy.mean(x, axis=(-2, -1)))
        x = self.head(x)
        return x


def _convnext_torch_weights_mapping(old_key, new_key):
    new_mapping = new_key
    if "downsample_layers" in old_key:
        if "0/0/bias" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "h -> 1 h 1 1"}
        elif "0/0/weight" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "b c h w-> h w c b"}
        elif "downsample_layers/0" not in old_key and "1/bias" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "h -> 1 h 1 1"}
        elif "downsample_layers/0" not in old_key and "1/w" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "b c h w -> h w c b"}
    elif "dwconv" in old_key:
        if "bias" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "h -> 1 h 1 1"}
        elif "weight" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "a 1 c d -> c d a"}
    elif "grn" in old_key:
        new_mapping = {"key_chain": new_key, "pattern": "1 1 1 h -> h"}
    return new_mapping


def convnext(size: str, pretrained=True):
    """Loads a ConvNeXt with specified size, optionally pretrained."""
    size_dict = {
        "tiny": ([3, 3, 9, 3], [96, 192, 384, 768]),
        "small": ([3, 3, 27, 3], [96, 192, 384, 768]),
        "base": ([3, 3, 27, 3], [128, 256, 512, 1024]),
        "large": ([3, 3, 27, 3], [192, 384, 768, 1536]),
    }
    try:
        depths, dims = size_dict[size]
    except KeyError:
        raise Exception("Enter a valid model size: tiny/small/base/large")

    if not pretrained:
        return ConvNeXt(version=1, depths=depths, dims=dims)

    weight_dl = {
        "tiny": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",  # noqa
        "small": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",  # noqa
        "base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",  # noqa
        "large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",  # noqa
    }

    reference_model = ConvNeXt(version=1, depths=depths, dims=dims)
    w_clean = ivy_models.helpers.load_torch_weights(
        weight_dl[size], reference_model, custom_mapping=_convnext_torch_weights_mapping
    )
    return ConvNeXt(version=1, depths=depths, dims=dims, v=w_clean)


def convnextv2(size: str, pretrained=True):
    """Loads a ConvNeXtV2 with specified size, optionally pretrained."""
    size_dict = {
        "atto": ([2, 2, 6, 2], [40, 80, 160, 320]),
        "base": ([3, 3, 27, 3], [128, 256, 512, 1024]),
    }
    try:
        depths, dims = size_dict[size]
    except KeyError:
        raise Exception("Enter a valid model size: tiny/small/base/large")

    if not pretrained:
        return ConvNeXt(version=2, depths=depths, dims=dims)

    weight_dl = {
        "atto": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt",  # noqa
        "base": "https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_base_1k_224_ema.pt",  # noqa
    }

    reference_model = ConvNeXt(version=2, depths=depths, dims=dims)
    w_clean = ivy_models.helpers.load_torch_weights(
        weight_dl[size], reference_model, custom_mapping=_convnext_torch_weights_mapping
    )
    return ConvNeXt(version=2, depths=depths, dims=dims, v=w_clean)
