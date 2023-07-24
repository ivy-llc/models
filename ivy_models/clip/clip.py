from typing import Tuple, Union

import numpy as np
import ivy
from ivy.stateful.initializers import Ones

from .layers import (
    CLIPModifiedResNet,
    CLIPTransformer,
    CLIPVisionTransformer,
    Embedding,
)
import ivy_models
from .misc import (
    get_model_args,
    get_clip_weights_url,
    load_clip_state_dict,
    tokenize,
    get_processors,
)

__all__ = ["CLIP", "clip", "tokenize", "get_processors"]


class CLIP(ivy.Module):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        # ivy
        device=None,
        v=None,
    ):
        """
        An ivy implementation of the CLIP model in fp32.
        The image encoders from the original implementation can be one of the following
        - Modified resnet variants (RN50, RN101, RN50x4, RN50x16, RNx64)
        - ViT variants: (ViT-B/32, ViT-B/16, ViT-L/14, ViT-l/14@336px)

        Parameters
        ----------
        embed_dim :
            Feature dimension that the text and image encoders will be projected to.
        image_resolution :
            Input image's resolution expected by the image encoder. (e.g. 224)
        vision layers :
            For the ViT image encoders it's the number of residual attention block.
            For the modified Resnets it's a tuple of four integers that represent the
            number of residual block in each of the four residual layers.
        vision_width :
            For the Resnets it's the number of channels in the first residual layer.
            For the ViT it's the transformer's feature dimension.
            (.i.e. In both cases the final visual features are projected to embed_dim.)
        vision_patch_size:
            The patch size of the ViT encoder. Not application to the Resnets.
        context_length :
            The context length of the text encoder
        vocab_size :
            The size of the vocabulary. Used in the embedding layer.
        transformer_width :
            The feature dimension of the text encoder.
            (e.i. It's later projected to embed_dim)
        transformer_heads :
            Number of attention head per residual attention block for the text encoder.
        transformer_layers :
            Number of residual attention block in the text encoder.
        """
        self.embed_dim = embed_dim
        self.image_resolution = image_resolution
        self.vision_layers = vision_layers
        self.vision_width = vision_width
        self.vision_patch_size = vision_patch_size

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers

        self._pos_embed_shape = (self.context_length, self.transformer_width)
        self._text_proj_shape = (self.transformer_width, self.embed_dim)
        self._scale_init = Ones()

        super().__init__(device=device, v=v)

    def _build(self, *args, **kwargs):
        if isinstance(self.vision_layers, (tuple, list)):
            vision_heads = self.vision_width * 32 // 64
            self.visual = CLIPModifiedResNet(
                layers=self.vision_layers,
                output_dim=self.embed_dim,
                heads=vision_heads,
                input_resolution=self.image_resolution,
                width=self.vision_width,
            )
        else:
            vision_heads = self.vision_width // 64
            self.visual = CLIPVisionTransformer(
                input_resolution=self.image_resolution,
                patch_size=self.vision_patch_size,
                width=self.vision_width,
                layers=self.vision_layers,
                heads=vision_heads,
                output_dim=self.embed_dim,
            )

        self.transformer = CLIPTransformer(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.token_embedding = Embedding(self.vocab_size, self.transformer_width)
        self.ln_final = ivy.LayerNorm([self.transformer_width])

    def _create_variables(self, *, device=None, dtype=None):
        v = {
            "positional_embedding": ivy.empty(
                self._pos_embed_shape, dtype=dtype, device=device
            ),
            "text_projection": ivy.empty(
                self._text_proj_shape, dtype=dtype, device=device
            ),
            # Casting to float32 because of an issue with avg_pool2d for jax backend
            # when jax_enable_x64 is set to True
            "logit_scale": self._scale_init.create_variables([], device, dtype=dtype)
            * np.log(1 / 0.07).astype(ivy.float32),
        }
        return v

    def build_attention_mask(self):
        # Create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask for floats; but ivy expect a boolean mask
        mask = ivy.ones((self.context_length, self.context_length))
        mask = mask.tril(k=0)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.v.w.dtype

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.v.positional_embedding
        x = x.permute_dims((1, 0, 2))  # NLD -> LND
        x = self.transformer(x)
        x = x.permute_dims((1, 0, 2))  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding
        # (eot_token is the highest number in each sequence)
        x = x[ivy.arange(x.shape[0]), text.argmax(axis=-1)] @ self.v.text_projection

        return x

    def _forward(
        self,
        image: Union[ivy.Array, ivy.NativeArray],
        text: Union[ivy.Array, ivy.NativeArray],
    ):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.vector_norm(
            axis=1, keepdims=True
        )
        text_features = text_features / text_features.vector_norm(axis=1, keepdims=True)

        # cosine similarity as logits
        logit_scale = self.v.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def _clip_torch_mapping(old_key, new_key):
    new_mapping = new_key

    if "conv" in old_key:
        if "/weight" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "o c h w -> h w c o "}
    if "downsample" in old_key:
        if "/0/weight" in old_key:
            new_mapping = {"key_chain": new_key, "pattern": "o c h w -> h w c o "}

    return new_mapping


def clip(name: str, pretrained=True):
    """
    Load a pretrained CLIP model variant.

    Parameters
    ----------
    name : str
        A model name listed in `clip.available_models()`.
        It's actually the pretrained image encoder that'll be used in the model.
        One in this list ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32',
        'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']

    Returns
    -------
    model : ivy.Module
        The pretrained CLIP model
    """
    url = get_clip_weights_url(name)
    state_dict = load_clip_state_dict(url)
    args = get_model_args(state_dict)
    model = CLIP(*args)

    if not pretrained:
        return model

    raw_keys_to_prune = [
        "context_length",
        "input_resolution",
        "vocab_size",
        "num_batches_tracked",
    ]
    clean_weights = ivy_models.helpers.load_torch_weights(
        url,
        model,
        raw_keys_to_prune=raw_keys_to_prune,
        custom_mapping=_clip_torch_mapping,
        jit=True,
        data_type=ivy.float32,
    )
    model = CLIP(*args, v=clean_weights)
    return model
