from typing import Union

import ivy


class Identity(ivy.Module):
    def __init__(self):
        super(Identity, self).__init__()
    
    def _forward(self, x):
        return x


class Embedding(ivy.Module):
    def __init__(self, vocab_size, embed_dim, max_norm=None, device=None, dtype=None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_norm = max_norm
        self.weight = ivy.random_normal(shape=(self.vocab_size, self.embed_dim), device=device, dtype=dtype)
        super(Embedding, self).__init__(device=device, dtype=dtype)
    
    def _create_variables(self, device=None, dtype=None):
        v = {'weight': self.weight}
        return v

    def _forward(self, x):
        return ivy.embedding(self.v.weight, x, max_norm=self.max_norm)


class Bottleneck(ivy.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = ivy.Conv2D(inplanes, planes, [1,1], (1,1), 0, with_bias=False, data_format="NCHW")
        self.bn1 = ivy.BatchNorm2D(planes, data_format="NCS")
        self.relu1 = ivy.ReLU()

        self.conv2 = ivy.Conv2D(planes, planes, [3, 3], (1, 1), 1, with_bias=False, data_format="NCHW")
        self.bn2 = ivy.BatchNorm2D(planes, data_format="NCS")
        self.relu2 = ivy.ReLU()

        self.avgpool = ivy.AvgPool2D(stride, stride, 0, data_format="NCHW") if stride > 1 else Identity()

        self.conv3 = ivy.Conv2D(planes, planes * self.expansion, [1, 1], (1, 1), 0, with_bias=False, data_format="NCHW")
        self.bn3 = ivy.BatchNorm2D(planes * self.expansion, data_format="NCS")
        self.relu3 = ivy.ReLU()

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = ivy.Sequential(*[ivy.AvgPool2D(stride, stride, 0, data_format="NCHW"),
                                            ivy.Conv2D(inplanes, planes * self.expansion, [1, 1], (1, 1), 0, with_bias=False, data_format="NCHW"),
                                            ivy.BatchNorm2D(planes * self.expansion, data_format="NCS")])

        super().__init__()

    def _forward(self, x: ivy.Array):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(ivy.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        self.positional_embedding = ivy.random_normal(shape=(spacial_dim ** 2 + 1, embed_dim)) / embed_dim ** 0.5
        self.k_proj = ivy.Linear(embed_dim, embed_dim)
        self.q_proj = ivy.Linear(embed_dim, embed_dim)
        self.v_proj = ivy.Linear(embed_dim, embed_dim)
        self.c_proj = ivy.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.dot_prod_scale = 1/ivy.sqrt(embed_dim//self.num_heads)
        super().__init__()

    def _create_variables(self, device=None, dtype=None):
        v = {"positional_embedding": self.positional_embedding}
        return v

    def _forward(self, x):
        x = x.flatten(start_dim=2).permute_dims((2, 0, 1))  # NCHW -> (HW)NC
        x = ivy.concat([x.mean(axis=0, keepdims=True), x], axis=0)  # (HW+1)NC
        x = x + self.v.positional_embedding[:, None, :]  # (HW+1)NC
        x = ivy.multi_head_attention(
            x[:1].permute_dims((1,0,2)), # Ivy expects the query in NLE, not LNE
            self.dot_prod_scale,
            self.num_heads,
            context=x.permute_dims((1,0,2)),
            to_q_fn=self.q_proj,
            to_kv_fn=lambda x, v=None: (self.k_proj(x), self.v_proj(x)),
            to_out_fn=self.c_proj,
        ) # N1C
        return x.squeeze(1) # N1C -> NC


class ModifiedResNet(ivy.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = ivy.Conv2D(3, width // 2, [3, 3], (2,2), 1, with_bias=False, data_format="NCHW")
        self.bn1 = ivy.BatchNorm2D(width // 2, data_format="NCS")
        self.relu1 = ivy.ReLU()
        self.conv2 = ivy.Conv2D(width // 2, width // 2, [3, 3], (1,1), 1, with_bias=False, data_format="NCHW")
        self.bn2 = ivy.BatchNorm2D(width // 2, data_format="NCS")
        self.relu2 = ivy.ReLU()
        self.conv3 = ivy.Conv2D(width // 2, width, [3, 3], (1,1), 1, with_bias=False, data_format="NCHW")
        self.bn3 = ivy.BatchNorm2D(width, data_format="NCS")
        self.relu3 = ivy.ReLU()
        self.avgpool = ivy.AvgPool2D(2, 2, 0, data_format="NCHW")

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)
        super().__init__()

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return ivy.Sequential(*layers)

    def _forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class QuickGELU(ivy.Module):
    def _forward(self, x: Union[ivy.Array, ivy.NativeArray]):
        return x * ivy.sigmoid(1.702 * x)


class ResidualAttentionBlock(ivy.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: Union[ivy.Array, ivy.NativeArray] = None):
        self.attn = ivy.MultiHeadAttention(d_model, num_heads=n_head)
        self.ln_1 = ivy.LayerNorm([d_model])
        self.mlp = ivy.Sequential(ivy.Linear(d_model, d_model * 4), QuickGELU(), ivy.Linear(d_model * 4, d_model))
        self.ln_2 = ivy.LayerNorm([d_model])
        self.attn_mask = attn_mask

        super().__init__()

    def attention(self, x: Union[ivy.Array, ivy.NativeArray]):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to_device(x.device)
        return self.attn(x, attention_mask=self.attn_mask)

    def _forward(self, x: Union[ivy.Array, ivy.NativeArray]):
        x = x.permute_dims((1,0,2)) # LND -> NLD : ivy's MultiHeadAtention layer expects NLD
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        x = x.permute_dims((1,0,2)) # NLD -> LND
        return x


class Transformer(ivy.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: Union[ivy.Array, ivy.NativeArray] = None):
        self.width = width
        self.layers = layers
        self.resblocks = ivy.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        super().__init__()

    def _forward(self, x: Union[ivy.Array, ivy.NativeArray]):
        return self.resblocks(x)


class VisionTransformer(ivy.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = ivy.Conv2D(3, width, [patch_size,]*2, (patch_size,)*2, 0, with_bias=False, data_format="NCHW")

        scale = width ** -0.5
        self.class_embedding = scale * ivy.random_normal(shape=width)
        self.positional_embedding = scale * ivy.random_normal(shape=((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = ivy.LayerNorm([width])

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = ivy.LayerNorm([width])
        self.proj = scale * ivy.random_normal(shape=(width, output_dim))
        super().__init__()
    
    def _create_variables(self, device=None, dtype=None):
        v = {
            'class_embedding': self.class_embedding,
            'positional_embedding': self.positional_embedding,
            'proj': self.proj,
        }
        return v

    def _forward(self, x: Union[ivy.Array, ivy.NativeArray]):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape((x.shape[0], x.shape[1], -1))  # shape = [*, width, grid ** 2]
        x = x.permute_dims((0, 2, 1))  # shape = [*, grid ** 2, width]
        x = ivy.concat([self.v.class_embedding + ivy.zeros((x.shape[0], 1, x.shape[-1]), dtype=x.dtype, device=x.device), x], axis=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.v.positional_embedding
        x = self.ln_pre(x)

        x = x.permute_dims((1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = x.permute_dims((1, 0, 2))  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.v.proj is not None:
            x = x @ self.v.proj

        return x
