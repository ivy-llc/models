import math
import ivy
from ivy_models.dino.utils import trunc_normal_
from ivy.stateful.initializers import Zeros, GlorotUniform
from ivy_models.vit.layers import partial


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + ivy.random_uniform(0,1, shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(ivy.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(ivy.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=ivy.GELU, drop=0.):
        super(Mlp, self).__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.act_layer = act_layer
        self.drop = drop

    def _build(self, *args, **kwargs):
        self.fc1 = ivy.Linear(self.in_features, self.hidden_features)
        self.act = self.act_layer()
        self.fc2 = ivy.Linear(self.hidden_features, self.out_features)
        self.drop = ivy.Dropout(self.drop)

    def _forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(ivy.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        self.dim = dim
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

    def _build(self, *args, **kwargs):
        self.qkv = ivy.Linear(self.dim, self.dim * 3, bias=self.qkv_bias)
        self.attn_drop = ivy.Dropout(self.attn_drop)
        self.proj = ivy.Linear(self.dim, self.dim)
        self.proj_drop = ivy.Dropout(self.proj_drop)

    def _forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(ivy.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=ivy.GELU, norm_layer=ivy.LayerNorm):
        # Additional attributes
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop = drop
        self.attn_drop = attn_drop
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        super(Block, self).__init__()

    def _build(self, *args, **kwargs):
        self.norm1 = self.norm_layer(self.dim)
        self.attn = Attention(
            self.dim, num_heads=self.num_heads, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale, attn_drop=self.attn_drop, proj_drop=self.drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else ivy.Identity()
        self.norm2 = self.norm_layer(self.dim)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=self.dim, hidden_features=mlp_hidden_dim, act_layer=self.act_layer, drop=self.drop)

    def _forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(ivy.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        super(PatchEmbed).__init__()

    def _build(self, *args, **kwargs):
        self.proj = ivy.Conv2D(self.in_chans, self.embed_dim, [self.patch_size, self.patch_size], self.patch_size, 0)

    def _forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(ivy.Module):
    """ Vision Transformer """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=ivy.LayerNorm, device=None, dtype=None, v: ivy.Container = None, **kwargs) -> None:
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token_shape = (1, 1, embed_dim)
        self.cls_token = Zeros()
        self.pos_embed_shape = (1, self.num_patches + 1, embed_dim)
        self.pos_embed = Zeros()
        self._weight_init = GlorotUniform()
        self._bias_init = Zeros()
        self.num_features = self.embed_dim = embed_dim
        self._w_shape = (embed_dim,)
        self._b_shape = (embed_dim,)
        super(VisionTransformer, self).__init__(v=v, device=device, dtype=dtype)

    def _build(self, *args, **kwargs):
        self.pos_drop = ivy.Dropout(prob=self.drop_rate)
        dpr = [x.item() for x in ivy.linspace(0, self.drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = [
            Block(
                dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer)
            for i in range(self.depth)]
        self.norm = self.norm_layer(self.embed_dim)

        # Classifier head
        self.head = ivy.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else ivy.Identity()

        # trunc_normal_(self.v.pos_embed, std=.02)
        # trunc_normal_(self.v.cls_token, std=.02)
        
        
    def _create_variables(self, *, device=None, dtype=None):
        # w = self._weight_init.create_variables(
        #     self._w_shape, device, dtype
        # )
        # v = {
        #     "w": trunc_normal_(w, std=.02),
        # }
        # v = dict(
        #     **v,
        #     b=self._b_init.create_variables(
        #         self._b_shape,
        #         device,
        #         dtype=dtype,
        #     ),
        # )
        v = {}
        v = dict(**v,
            class_token= self.cls_token.create_variables(
                self.cls_token_shape, device, dtype=dtype
            ))
        v = dict(**v, pos_embed= self.pos_embed.create_variables(self.pos_embed_shape, device, dtype=dtype))
        return v

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.v.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.v.pos_embed
        class_pos_embed = self.v.pos_embed[:, 0]
        patch_pos_embed = self.v.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = ivy.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return ivy.concat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = ivy.expand(self.v.cls_token, (B,-1,-1))
        x = ivy.concat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def _forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(ivy.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == "__main__":
    model = vit_tiny()
