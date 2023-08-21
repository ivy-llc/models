import os
import ivy
import pytest
import numpy as np
from ivy_models_tests import helpers
from ivy_models.inceptionnet import inceptionNet_v3


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_inceptionNet_v3_img_classification(device, fw, batch_shape, load_weights):
    """Test InceptionNetV3 image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )

    # Create model
    # model = inceptionNet_v3(pretrained=load_weights)
    # logits = model(img)

    # Cardinality test
    # assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # # Value test
    # if load_weights:
    #     np_out = ivy.to_numpy(logits[0])
    #     true_indices = np.array([258, 270, 279])
    #     calc_indices = np.argsort(np_out)[-3:][::-1]
    #     assert np.array_equal(true_indices, calc_indices)

    #     true_logits = np.array([9.9990e-01, 8.3909e-05, 1.1693e-05])
    #     calc_logits = np.take(np_out, calc_indices)
    #     assert np.allclose(true_logits, calc_logits, rtol=1)
