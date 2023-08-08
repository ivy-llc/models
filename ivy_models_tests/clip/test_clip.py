import os
import random

import ivy
import numpy as np
from PIL import Image

from ivy_models import clip, get_processors


VARIANTS_LOGITS = {
    "RN50": ivy.array([12.2088346, 14.9655876, 21.0058422]),
    "RN101": ivy.array([35.2729797, 36.0812988, 42.8816681]),
    "RN50x4": ivy.array([27.0689335, 29.3602104, 35.6379929]),
    "RN50x16": ivy.array([16.3668022, 20.4796104, 27.0634518]),
    "RN50x64": ivy.array([8.9237432, 14.1180887, 20.0675087]),
    "ViT-B/32": ivy.array([17.3713417, 19.5949516, 25.5068512]),
    "ViT-B/16": ivy.array([17.9823151, 20.7719479, 26.9038792]),
    "ViT-L/14": ivy.array([11.9254637, 15.6604385, 22.6723843]),
    "ViT-L/14@336px": ivy.array([10.9720955, 13.5543489, 21.4815979]),
}

load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS_LOGITS.keys()))
model = clip(model_var, pretrained=load_weights)
v = ivy.to_numpy(model.v)


def test_all_clip_img_classification(
    device,
    f,
    fw,
):
    """Test one CLIP variant for zero shot image classification."""
    image_name = "cat.jpg"
    one_shot_labels = ["a diagram", "a dog", "a cat"]
    batch_shape = [1]
    num_classes = len(one_shot_labels)
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image and processors
    tokenize, im_tfms = get_processors(model)
    img = Image.open(os.path.join(this_dir, "..", "..", "images", image_name))
    img = ivy.expand_dims(im_tfms(img), axis=0)
    text = tokenize(one_shot_labels)

    # Get logits and probs
    model.v = ivy.asarray(v)
    logits_per_image, logits_per_text = model(img, text)
    calc_probs = ivy.to_numpy(logits_per_image.softmax(axis=-1)[0])
    true_probs = ivy.to_numpy(VARIANTS_LOGITS[model_var].softmax())

    # Cardinality test
    assert logits_per_image.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    # Probs instead of logits because the raw weights are in fp16 and we used float32.
    if load_weights:
        assert np.allclose(true_probs, calc_probs, atol=5e-3)
