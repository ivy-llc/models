import os
import ivy
import pytest
import numpy as np
import random
from ivy_models import unet_carvana
from ivy_models_tests import helpers


load_weights = random.choice([True, False])
# Create model
model = unet_carvana(pretrained=load_weights)
v = ivy.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_unet_img_segmentation(device, fw, data_format):
    """Test UNet image segmentation"""
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "car.jpg"),
        256,
        224,
        data_format=data_format,
    )

    # Perform inference
    model.v = ivy.asarray(v)
    output = model(img, data_format=data_format)
    output_np = ivy.to_numpy(output)

    # Cardinality test
    assert output.shape == tuple([1, 224, 224, 2])

    if load_weights:
        assert np.allclose(output_np.sum(), np.array([111573.26]), rtol=1.0)
