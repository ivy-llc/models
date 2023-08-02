import os
import ivy
import pytest
import numpy as np

from ivy_models.unet import unet_carvana
from ivy_models_tests import helpers


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_unet_img_segmentation(device, fw, batch_shape, load_weights):
    """Test UNet image segmentation"""
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "car.jpg"), 256, 224
    )

    # Create model
    model = unet_carvana(pretrained=load_weights)

    # Perform inference
    output = model(img)
    output_np = ivy.to_numpy(output)

    # Cardinality test
    assert output.shape == tuple([1, 224, 224, 2])

    if load_weights:
        assert np.allclose(output_np.sum(), np.array([111573.26]), rtol=1.0)
