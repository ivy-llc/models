import os
import numpy as np
import pytest
import random
import ivy
from ivy_models_tests import helpers
from ivy_models import squeezenet1_0, squeezenet1_1


VARIANTS = {
    "squeezenet1_0": squeezenet1_0,
    "squeezenet1_1": squeezenet1_1,
}

load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS.keys()))
model = VARIANTS[model_var](dropout=0, pretrained=load_weights)
v = ivy.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_squeezenet_img_classification(device, fw, data_format):
    """Test SqueezeNet image classification."""
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

    # Create model
    model.v = ivy.asarray(v)
    logits = model(img, data_format=data_format)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.sort(np.array([282, 281, 285, 287]))
        calc_indices = np.sort(np.argsort(np_out)[-5:][::-1])
        assert np.array_equal(true_indices, calc_indices[:4])
