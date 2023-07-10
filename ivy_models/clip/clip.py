from typing import Tuple, Union

import numpy as np
import ivy
from ivy.stateful.initializers import Zeros, Ones

from .layers import *
from .misc import get_model_args, get_ivy_weights, load_clip_state_dict, tokenize, get_processors

__all__ = ["CLIP", "load_clip", "tokenize", "get_processors"]

class CLIP(ivy.Module):
    def __init__(self,
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
                 #ivy
                 device=None,
                 v=None
                 ):

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
        self._pos_embed_init = Zeros()
        self._text_proj_shape = (self.transformer_width, self.embed_dim)
        self._text_proj_init = Zeros()
        self._scale_init = Ones()

        super().__init__(device=device, v=v)
    
    def _build(self, *args, **kwargs):
        if isinstance(self.vision_layers, (tuple, list)):
            vision_heads = self.vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=self.vision_layers,
                output_dim=self.embed_dim,
                heads=vision_heads,
                input_resolution=self.image_resolution,
                width=self.vision_width
            )
        else:
            vision_heads = self.vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=self.image_resolution,
                patch_size=self.vision_patch_size,
                width=self.vision_width,
                layers=self.vision_layers,
                heads=vision_heads,
                output_dim=self.embed_dim
            )

        self.transformer = Transformer(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.token_embedding = Embedding(self.vocab_size, self.transformer_width)
        self.ln_final = ivy.LayerNorm([self.transformer_width])


    def _create_variables(self, *, device=None, dtype=None):
        v = {
            'positional_embedding': self._pos_embed_init.create_variables(self._pos_embed_shape, device, dtype=dtype),
            'text_projection' : self._text_proj_init.create_variables(self._text_proj_shape, device, dtype=dtype),
            # Casting to float32 because of an issue with avg_pool2d for jax backend when jax_enable_x64 is set to True
            'logit_scale' : self._scale_init.create_variables([], device, dtype=dtype) * np.log(1 / 0.07).astype(ivy.float32),
        }
        return v

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; but ivy expect a boolean mask (it's converted to a boolean mask)
        # IVY: Made changes to the mask cause ivy's behavior for float masks is different compared to torch
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
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[ivy.arange(x.shape[0]), text.argmax(axis=-1)] @ self.v.text_projection

        return x

    def _forward(self, image: Union[ivy.Array, ivy.NativeArray], text: Union[ivy.Array, ivy.NativeArray]):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.vector_norm(axis=1, keepdims=True)
        text_features = text_features / text_features.vector_norm(axis=1, keepdims=True)

        # cosine similarity as logits
        logit_scale = self.v.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def load_clip(name: str, pretrained=True):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    Returns
    -------
    model : ivy.Module
        The CLIP model
    """
    state_dict = load_clip_state_dict(name)
    args = get_model_args(state_dict)
    model = CLIP(*args)

    if not pretrained:
        return model
    
    clean_weights = get_ivy_weights(model.v, state_dict)
    model = CLIP(*args, v=clean_weights)
    return model
