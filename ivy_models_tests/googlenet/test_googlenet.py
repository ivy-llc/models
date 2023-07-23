import os
import ivy
import pytest
import numpy as np

from ivy_models.googlenet import inceptionNet_v1
from ivy_models_tests import helpers

def np_softmax(inputs):
    """apply the softmax on the output"""
    return np.exp(inputs) / np.sum(np.exp(inputs))

@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_inception_v1_img_classification(device, f, fw, batch_shape, load_weights):
    """Test Inception-V1 image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

     # Load image
    img = ivy.asarray(
        helpers.load_and_preprocess_img(
            os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
        ),
    )

    # Create model
    model = inceptionNet_v1(pretrained=load_weights)

    # Perform inference
    output, _, _ = model(img)

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        output = output[0]
        true_indices = ivy.array([282, 281, 285])
        calc_indices = ivy.argsort(output, descending=True)[:3]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([0.2539, 0.2391, 0.1189])
        calc_logits = np.take(
            np_softmax(ivy.to_numpy(output)), ivy.to_numpy(calc_indices)
        )

        assert np.allclose(true_logits, calc_logits, rtol=0.5)

