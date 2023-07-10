from typing import Tuple, Union

import numpy as np
import ivy

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

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = Embedding(vocab_size, transformer_width)
        self.positional_embedding = ivy.empty((self.context_length, transformer_width))
        self.ln_final = ivy.LayerNorm([transformer_width])

        self.text_projection = ivy.empty((transformer_width, embed_dim))
        # Casting to float32 because of an issue with avg_pool2d for jax backend when jax_enable_x64 is set to True
        self.logit_scale = ivy.ones([]) * np.log(1 / 0.07).astype(ivy.float32)

        super().__init__(device=device, v=v)
    
    def _create_variables(self, *, device=None, dtype=None):
        v = {
            'positional_embedding': self.positional_embedding,
            'text_projection' : self.text_projection,
            'logit_scale' : self.logit_scale,
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
