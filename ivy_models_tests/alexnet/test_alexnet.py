import os
import random
import ivy
import pytest
import numpy as np

from ivy_models.alexnet import alexnet
from ivy_models_tests import helpers

import jax

jax.config.update("jax_enable_x64", False)

load_weights = random.choice([False, True])
model = alexnet(pretrained=load_weights)
v = ivy.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_alexnet_tiny_img_classification(device, f, fw, data_format):
    """Test AlexNet image classification."""
    num_classes = 1000
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"),
        256,
        224,
        data_format=data_format,
        to_ivy=True,
    )

    model.v = ivy.asarray(v)
    logits = model(img, data_format=data_format)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 287, 896])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([23.5786, 22.791977, 20.917543, 19.49762, 16.102253])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)
