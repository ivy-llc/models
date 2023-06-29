import os
import ivy
import pytest
import numpy as np

from ivy_models.resnet import resnet_18
from ivy_models_tests import helpers


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_resnet_18_img_classification(device, f, fw, batch_shape, load_weights):
    """Test ResNet-18 image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )

    # Create model
    model = resnet_18(pretrained=load_weights)

    # Perform inference
    output = model(img)

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        output = output[0]
        true_indices = ivy.array([282, 285, 281]).sort()
        calc_indices = ivy.argsort(output, descending=True)[:3].sort()

        assert np.array_equal(true_indices, calc_indices)

        true_logits = ivy.array([13.348762, 12.343024, 11.933505])
        calc_logits = ivy.take_along_axis(output, calc_indices, 0)

        assert np.allclose(true_logits, calc_logits, rtol=0.5)
