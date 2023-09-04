import os
import random
import ivy
import pytest
import numpy as np
from ivy_models_tests import helpers
from ivy_models.inceptionnet import inceptionNet_v3
import jax

jax.config.update("jax_enable_x64", False)

load_weights = random.choice([False, True])
model = inceptionNet_v3(pretrained=load_weights)
v = ivy.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_inceptionNet_v3_img_classification(device, fw, data_format):
    """Test InceptionNetV3 image classification."""
    num_classes = 1000
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "dog.jpg"),
        256,
        224,
        data_format=data_format,
        to_ivy=True,
    )

    # Create model
    model.v = ivy.asarray(v)
    logits, _ = model(img, data_format=data_format)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([258, 270, 279])
        calc_indices = np.argsort(np_out)[-3:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([9.9990e-01, 8.3909e-05, 1.1693e-05])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1)
