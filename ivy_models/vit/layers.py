from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union, Sequence
import collections
from itertools import repeat
from functools import partial
from ivy.stateful.initializers import Zeros
import ivy


def _make_ntuple(x: Any, n: int) -> Tuple[Any, ...]:
    """
    Make n-tuple from input x. If x is an iterable, then we just convert it to tuple.
    Otherwise, we will make a tuple of length n, all with value of x.

    Args:
        x (Any): input value
        n (int): length of the resulting tuple
    """
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(repeat(x, n))


class ConvNormActivation(ivy.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., ivy.Module]] = ivy.BatchNorm2D,
        activation_layer: Optional[Callable[..., ivy.Module]] = ivy.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = False,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., ivy.Module] = ivy.Conv2D,
    ) -> None:
        if padding is None:
            if isinstance(kernel_size, int) and isinstance(dilation, int):
                padding = (kernel_size - 1) // 2 * dilation
            else:
                _conv_dim = (
                    len(kernel_size)
                    if isinstance(kernel_size, Sequence)
                    else len(dilation)
                )
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple(
                    (kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim)
                )
        if bias is None:
            bias = norm_layer is None

        layers = [
            conv_layer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            )
        ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels))

        if activation_layer is not None:
            params = {} if inplace is None else {"inplace": inplace}
            layers.append(activation_layer(**params))
        self.out_channels = out_channels

        super().__init__(*layers)

        if self.__class__ == ConvNormActivation:
            ivy.warnings.warn(
                "Don't use ConvNormActivation directly, please use Conv2dNormActivation instead."
            )


class Conv2dNormActivation(ConvNormActivation):
    """
    Configurable block used for Convolution2d-Normalization-Activation blocks.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the Convolution-Normalization-Activation block
        kernel_size: (int, optional): Size of the convolving kernel. Default: 3
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: None, in which case it will be calculated as ``padding = (kernel_size - 1) // 2 * dilation``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        norm_layer (Callable[..., ivy.Module], optional): Norm layer that will be stacked on top of the convolution layer. If ``None`` this layer won't be used. Default: ``ivy.BatchNorm2D``
        activation_layer (Callable[..., ivy.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the conv layer. If ``None`` this layer won't be used. Default: ``ivy.ReLU``
        dilation (int): Spacing between kernel elements. Default: 1
        inplace (bool): Parameter for the activation layer, which can optionally do the operation in-place. Default ``True``
        bias (bool, optional): Whether to use bias in the convolution layer. By default, biases are included if ``norm_layer is None``.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., ivy.Module]] = ivy.BatchNorm2D,
        activation_layer: Optional[Callable[..., ivy.Module]] = ivy.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            ivy.Conv2D,
        )


class MLP(ivy.Sequential):
    """This block implements the multi-layer perceptron (MLP) module.

    Args:
        in_channels (int): Number of channels of the input
        hidden_channels (List[int]): List of the hidden channel dimensions
        norm_layer (Callable[..., ivy.Module], optional): Norm layer that will be stacked on top of the linear layer. If ``None`` this layer won't be used. Default: ``None``
        activation_layer (Callable[..., ivy.Module], optional): Activation function which will be stacked on top of the normalization layer (if not None), otherwise on top of the linear layer. If ``None`` this layer won't be used. Default: ``ivy.ReLU``
        inplace (bool, optional): Parameter for the activation layer, which can optionally do the operation in-place.
            Default is ``None``, which uses the respective default values of the ``activation_layer`` and Dropout layer.
        bias (bool): Whether to use bias in the linear layer. Default ``True``
        dropout (float): The probability for the dropout layer. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: List[int],
        norm_layer: Optional[Callable[..., ivy.Module]] = None,
        activation_layer: Optional[Callable[..., ivy.Module]] = ivy.ReLU,
        inplace: Optional[bool] = None,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        params = {} if inplace is None else {"inplace": inplace}

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(ivy.Linear(in_dim, hidden_dim, with_bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(ivy.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(ivy.Linear(in_dim, hidden_channels[-1], with_bias=bias))
        layers.append(ivy.Dropout(dropout, **params))

        super().__init__(*layers)


class ConvStemConfig(NamedTuple):
    out_channels: int
    kernel_size: int
    stride: int
    norm_layer: Callable[..., ivy.Module] = ivy.BatchNorm2D
    activation_layer: Callable[..., ivy.Module] = ivy.ReLU


class VIT_MLPBlock(MLP):
    """Transformer MLP block."""

    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(
            in_dim,
            [mlp_dim, in_dim],
            activation_layer=ivy.GELU,
            inplace=None,
            dropout=dropout,
        )


class VIT_EncoderBlock(ivy.Module):
    """Transformer encoder block."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., ivy.Module] = partial(ivy.LayerNorm, eps=1e-6),
    ):
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout_p = dropout
        self.attention_dropout = attention_dropout
        self.norm_layer = norm_layer

        super().__init__()

    def _build(self, *args, **kwargs) -> bool:
        # Attention block
        self.ln_1 = self.norm_layer(self.hidden_dim)
        self.self_attention = ivy.MultiHeadAttention(
            self.hidden_dim,
            num_heads=self.num_heads,
            dropout_rate=self.attention_dropout,
        )
        self.dropout = ivy.Dropout(self.dropout_p)

        # MLP block
        self.ln_2 = self.norm_layer(self.hidden_dim)
        self.mlp = VIT_MLPBlock(self.hidden_dim, self.mlp_dim, self.dropout_p)

    def _forward(self, input):
        ivy.utils.assertions.check_true(
            input.get_num_dims() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        x = self.ln_1(input)
        x = self.self_attention(x, x, x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class VIT_Encoder(ivy.Module):
    """Transformer Model Encoder for sequence to sequence translation."""

    def __init__(
        self,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., ivy.Module] = partial(ivy.LayerNorm, eps=1e-6),
    ):
        # Note that batch_size is on the first dim because
        # we have batch_first=True in nn.MultiAttention() by default
        self._pos_embedding_shape = (1, seq_length, hidden_dim)
        self.pos_embedding = Zeros()  # from BERT
        self.dropout = ivy.Dropout(dropout)
        layers = []
        for i in range(num_layers):
            layers.append(VIT_EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            ))
        self.layers = ivy.Sequential(
            *layers
        )
        self.ln = norm_layer(hidden_dim)
        super().__init__()

    def _create_variables(self, device, dtype=None):
        return {
            "pos_embedding": self.pos_embedding.create_variables(
                self._pos_embedding_shape, device, dtype=dtype
            )
        }

    def _forward(self, input):
        ivy.utils.assertions.check_true(
            input.get_num_dims() == 3,
            f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}",
        )
        input = input + self.v.pos_embedding
        return self.ln(self.layers(self.dropout(input)))
