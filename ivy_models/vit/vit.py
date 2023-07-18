import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, NamedTuple, Optional, Tuple, Union, Sequence
import collections
from itertools import repeat


import ivy
from ivy_models.helpers import load_torch_weights

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
                _conv_dim = len(kernel_size) if isinstance(kernel_size, Sequence) else len(dilation)
                kernel_size = _make_ntuple(kernel_size, _conv_dim)
                dilation = _make_ntuple(dilation, _conv_dim)
                padding = tuple((kernel_size[i] - 1) // 2 * dilation[i] for i in range(_conv_dim))
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


class MLPBlock(MLP):
    """Transformer MLP block."""
    def __init__(self, in_dim: int, mlp_dim: int, dropout: float):
        super().__init__(in_dim, [mlp_dim, in_dim], activation_layer=ivy.GELU, inplace=None, dropout=dropout)


class EncoderBlock(ivy.Module):
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

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = ivy.MultiHeadAttention(hidden_dim, num_heads=num_heads, dropout_rate=attention_dropout)
        self.dropout = ivy.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)
        super().__init__()

    def _forward(self, input):
        ivy.utils.assertions.check_true(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


class Encoder(ivy.Module):
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
        self.pos_embedding = ivy.empty((1, seq_length, hidden_dim))  # from BERT
        self.dropout = ivy.Dropout(dropout)
        layers: OrderedDict[str, ivy.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = ivy.Sequential(layers)
        self.ln = norm_layer(hidden_dim)
        super().__init__()

    def _forward(self, input):
        ivy.utils.assertions.check_true(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        input = input + self.pos_embedding
        return self.ln(self.layers(self.dropout(input)))


class VisionTransformer(ivy.Module):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        num_classes: int = 1000,
        representation_size: Optional[int] = None,
        norm_layer: Callable[..., ivy.Module] = partial(ivy.LayerNorm, eps=1e-6),
        conv_stem_configs: Optional[List[ConvStemConfig]] = None,
        v=None
    ):
        ivy.utils.assertions.check_true(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        if conv_stem_configs is not None:
            # As per https://arxiv.org/abs/2106.14881
            seq_proj = OrderedDict()
            prev_channels = 3
            for i, conv_stem_layer_config in enumerate(conv_stem_configs):
                seq_proj[f"conv_bn_relu_{i}"] = Conv2dNormActivation(
                        in_channels=prev_channels,
                        out_channels=conv_stem_layer_config.out_channels,
                        kernel_size=conv_stem_layer_config.kernel_size,
                        stride=conv_stem_layer_config.stride,
                        norm_layer=conv_stem_layer_config.norm_layer,
                        activation_layer=conv_stem_layer_config.activation_layer,
                    )
                prev_channels = conv_stem_layer_config.out_channels
            seq_proj["conv_last"] = ivy.Conv2D(prev_channels, hidden_dim, [1, 1], 1, 0)
            self.conv_proj: ivy.Module = ivy.Sequential(seq_proj)
        else:
            self.conv_proj = ivy.Conv2D(
                3, hidden_dim, [patch_size, patch_size], patch_size, 0
            )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = ivy.zeros(shape=(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length

        heads_layers: OrderedDict[str, ivy.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = ivy.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = ivy.Linear(hidden_dim, representation_size)
            heads_layers["act"] = ivy.tanh()
            heads_layers["head"] = ivy.Linear(representation_size, num_classes)

        self.heads = ivy.Sequential(heads_layers)
        super().__init__(v=v)


    def _process_input(self, x):
        n, c, h, w = x.shape
        p = self.patch_size
        ivy.utils.assertions.check_true(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        ivy.utils.assertions.check_true(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def _forward(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = ivy.concat([batch_class_token, x], axis=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    v=None
) -> VisionTransformer:
    
    model = VisionTransformer(
        image_size=224,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        v=v,
    )

    return model


def vit_b_16(v=None, pretrained=True) -> VisionTransformer:
    ref_model = _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072
    )
    if pretrained:
        url = 'https://download.pytorch.org/models/vit_b_16-c867db91.pth'
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=None,
        )
        
    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        v=w_clean,
    )

def vit_b_32(v=None, pretrained=True) -> VisionTransformer:
    ref_model = _vision_transformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072
    )
    if pretrained:
        url = 'https://download.pytorch.org/models/vit_b_32-d86f8d99.pth'
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=None,
        )
        
    return _vision_transformer(
        patch_size=32,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        v=w_clean,
    )
    

def vit_l_16(v=None, pretrained=True) -> VisionTransformer:
    ref_model = _vision_transformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096
    )
    if pretrained:
        url = 'https://download.pytorch.org/models/vit_l_16-852ce7e3.pth'
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=None,
        )
        
    return _vision_transformer(
        patch_size=16,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        v=w_clean,
    )
    

def vit_l_32(v=None, pretrained=True) -> VisionTransformer:
    ref_model = _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096
    )
    if pretrained:
        url = 'https://download.pytorch.org/models/vit_l_32-c7638314.pth'
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=None,
        )
        
    return _vision_transformer(
        patch_size=32,
        num_layers=24,
        num_heads=16,
        hidden_dim=1024,
        mlp_dim=4096,
        v=w_clean,
    )

def vit_h_14(v=None, pretrained=True) -> VisionTransformer:
    ref_model = _vision_transformer(
        patch_size=14,
        num_layers=12,
        num_heads=14,
        hidden_dim=768,
        mlp_dim=3072
    )
    if pretrained:
        url = 'https://download.pytorch.org/models/vit_h_14_swag-80465313.pth'
        w_clean = load_torch_weights(
            url,
            ref_model,
            raw_keys_to_prune=["num_batches_tracked"],
            custom_mapping=None,
        )
        
    return _vision_transformer(
        patch_size=14,
        num_layers=12,
        num_heads=14,
        hidden_dim=768,
        mlp_dim=3072,
        v=w_clean,
    )


if __name__ == '__main__':
    ivy.set_torch_backend()
    # model = VisionTransformer(image_size=224, patch_size=16, num_layers=12, num_heads=12,hidden_dim=768, mlp_dim=3072)
    model = vit_b_16()
    print(model.v)