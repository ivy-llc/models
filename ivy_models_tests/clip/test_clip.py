import os

import ivy
import pytest
import numpy as np
from PIL import Image

from ivy_models.clip import load_clip, get_processors


@pytest.mark.parametrize(
    "image_encoder",
    [
        "RN50",
        "RN101",
        "RN50x4",
        "RN50x16",
        "RN50x64",
        "ViT-B/32",
        "ViT-B/16",
        "ViT-L/14",
        "ViT-L/14@336px",
    ],
)
@pytest.mark.parametrize("image_name", ["cat.jpg"])
@pytest.mark.parametrize("text_input", [["a diagram", "a dog", "a cat"]])
@pytest.mark.parametrize("batch_shape", [[1]])
def test_all_clip_img_classification(
    device, f, fw, image_encoder, image_name, text_input, batch_shape
):
    """Test all CLIP variants for zero shot image classification."""
    true_logits = {
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

    num_classes = len(text_input)
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Create model
    model = load_clip(image_encoder)

    # Load image and processors
    tokenize, im_tfms = get_processors(model)
    img = Image.open(os.path.join(this_dir, "..", "..", "images", image_name))
    img = ivy.expand_dims(im_tfms(img), axis=0)
    text = tokenize(text_input)

    # Get logits and probs
    logits_per_image, logits_per_text = model(img, text)
    calc_probs = logits_per_image.softmax(axis=-1)[0]
    true_probs = true_logits[image_encoder].softmax()

    # Cardinality test
    print("Cardinality: ", logits_per_image.shape)
    assert logits_per_image.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    # using probs instead of logits because the original torch model is in fp16, but in float32 in ivy
    print("Calc logits : ", calc_probs)
    assert np.allclose(true_probs, calc_probs, atol=1e-3)
