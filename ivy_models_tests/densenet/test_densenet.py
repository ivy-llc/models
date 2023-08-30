import os
import random
import ivy
import pytest
import numpy as np

from ivy_models.densenet import densenet121, densenet161, densenet169, densenet201
from ivy_models_tests import helpers


VARIANTS = {
    "densenet121": densenet121,
    "densenet161": densenet161,
    "densenet169": densenet169,
    "densenet201": densenet201,
}


LOGITS = {
    "densenet121": [8.791083, 6.803193, 5.147233, 2.5118146, 1.3056283],
    "densenet161": [8.467648, 8.057183, 6.881177, 2.6506257, 1.8245339],
    "densenet169": [8.707129, 7.919885, 5.577528, 2.378178, 2.0281594],
    "densenet201": [8.77628, 7.687718, 6.09846, 2.25323, 2.2160888],
}

load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS.keys()))
model = VARIANTS[model_var](pretrained=load_weights)
v = ivy.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_densenet_img_classification(device, fw, data_format):
    """Test DenseNet image classification."""
    num_classes = 1000
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        helpers.load_and_preprocess_img(
            os.path.join(this_dir, "..", "..", "images", "cat.jpg"),
            256,
            224,
            data_format="NHWC",
            to_ivy=True,
        ),
    )

    # Create model
    model.v = ivy.asarray(v)
    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 287, 292])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)
