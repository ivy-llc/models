import ivy
from ivy.stateful.initializers import Zeros, Ones, Constant
from ivy.stateful.module import Module

class ConvNeXtBlock(ivy.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        self.dim = dim
        self.layer_scale_init_value = layer_scale_init_value
        self.drop_path = lambda x: x
        super(ConvNeXtBlock, self).__init__()

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


class ConvNeXtLayerNorm(ivy.Module):
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
