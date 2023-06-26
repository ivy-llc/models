import ivy
import ivy_models
from ivy.stateful.module import Module
from ivy.stateful.initializers import Zeros, Ones, Constant


class Block(ivy.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path = lambda x: x
        super(Block, self).__init__()

    def _build(self, *args, **kwargs):
        self.dwconv = ivy.DepthwiseConv2D(
            self.dim, [7, 7], (1, 1), 3, data_format="NCHW"
        )
        self.norm = ivy.LayerNorm([self.dim], eps=1e-6)
        self.pwconv1 = ivy.Linear(self.dim, 4 * self.dim)
        self.act = ivy.GELU()
        self.pwconv2 = ivy.Linear(4 * self.dim, self.dim)
        if self.layer_scale_init_value > 0:
            self.have_gamma = True
            self._gamma_init = Constant(self.layer_scale_init_value)
            self._gamma_shape = (self.dim,)

    def _create_variables(self, device, dtype=None):
        """Create internal variables for the layer."""
        if self.have_gamma:
            return {
                "gamma_param": self._gamma_init.create_variables(
                    self._gamma_shape, device, dtype
                ),
            }
        return {}

    def _forward(self, input):
        x = input
        x = self.dwconv(x)
        x = ivy.permute_dims(x, axes=(0, 2, 3, 1))
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.have_gamma:
            x = self.v.gamma_param * x
        x = ivy.permute_dims(x, axes=(0, 3, 1, 2))
        x = input + self.drop_path(x)
        return x


class ConvNeXt(ivy.Module):
    def __init__(
        self,
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

        super(ConvNeXt, self).__init__(device=device, v=v)

    def _build(self, *args, **kwargs):
        self.downsample_layers = []
        stem = ivy.Sequential(
            ivy.Conv2D(
                self.in_channels, self.dims[0], [4, 4], (4, 4), 0, data_format="NCHW"
            ),
            LayerNorm(self.dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = ivy.Sequential(
                LayerNorm(self.dims[i], eps=1e-6, data_format="channels_first"),
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
                    Block(
                        dim=self.dims[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=self.layer_scale_init_value,
                    )
                    for j in range(self.depths[i])
                ]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        self.norm = ivy.LayerNorm([self.dims[-1]], eps=1e-6)
        self.head = ivy.Linear(self.dims[-1], self.num_classes)

    def _forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm(ivy.mean(x, axis=(-2, -1)))
        x = self.head(x)
        return x


class LayerNorm(ivy.Module):
    """custom layernorm which can be applied to channel index 1."""

    def __init__(self, num_channels, data_format, eps=1e-6, device=None, dtype=None):
        if data_format == "channels_first":
            self.channel_index = 1
        else:
            raise NotImplementedError

        self.eps = eps
        normalized_shape = (num_channels,)
        self._weight_shape = normalized_shape
        self._bias_shape = normalized_shape
        self._weight_init = Ones()
        self._bias_init = Zeros()
        Module.__init__(self, device=device, dtype=dtype)

    def _create_variables(self, device, dtype=None):
        return {
            "weight": self._weight_init.create_variables(
                self._weight_shape, device, dtype=dtype
            ),
            "bias": self._bias_init.create_variables(
                self._bias_shape, device, dtype=dtype
            ),
        }

    def _forward(self, x):
        u = ivy.mean(x, axis=self.channel_index, keepdims=True)
        s = ivy.mean((x - u) ** 2, axis=self.channel_index, keepdims=True)
        x = (x - u) / ivy.sqrt(s + self.eps)
        return self.v.weight[:, None, None] * x + self.v.bias[:, None, None]


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
        return ConvNeXt(depths=depths, dims=dims)

    weight_dl = {
        "tiny": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",  # noqa
        "small": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",  # noqa
        "base": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",  # noqa
        "large": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",  # noqa
    }

    reference_model = ConvNeXt(depths=depths, dims=dims)
    w_clean = ivy_models.helpers.load_torch_weights(
        weight_dl[size], reference_model, custom_mapping=_convnext_torch_weights_mapping
    )
    reference_model.v = w_clean
    return reference_model
