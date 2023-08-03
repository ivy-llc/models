import os
import numpy as np
import pytest
import random
import ivy
from ivy_models_tests import helpers
from ivy_models import (
    convnext_tiny,
    convnext_small,
    convnext_base,
    convnext_large,
    convnextv2_atto,
    convnextv2_base,
)


VARIANTS = {
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
    "convnext_large": convnext_large,
    "convnextv2_atto": convnextv2_atto,
    "convnextv2_base": convnextv2_base,
}


LOGITS = {
    "convnext_tiny": [8.791083, 6.803193, 5.147233, 2.5118146, 1.3056283],
    "convnext_small": [8.467648, 8.057183, 6.881177, 2.6506257, 1.8245339],
    "convnext_base": [8.707129, 7.919885, 5.577528, 2.378178, 2.0281594],
    "convnext_large": [8.77628, 7.687718, 6.09846, 2.25323, 2.2160888],
    "convnextv2_atto": [7.1058, 6.6685, 5.9932, 2.6573, 2.2070],
    "convnextv2_base": [8.5643, 7.6972, 5.9340, 2.7507, 2.3775],
}

load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS.keys()))
model = VARIANTS[model_var](pretrained=load_weights)
v = ivy.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_convnext_img_classification(device, fw, data_format):
    """Test ConvNeXt image classification."""
    num_classes = 1000
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        helpers.load_and_preprocess_img(
            os.path.join(this_dir, "..", "..", "images", "cat.jpg"),
            256,
            224,
            data_format=data_format,
            to_ivy=True,
        ),
    )

    # Create model
    model.v = ivy.asarray(v)
    logits = model(img, data_format=data_format)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 287, 292])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = LOGITS[model_var]
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)
