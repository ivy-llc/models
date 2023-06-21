import os
import ivy
import pytest
import numpy as np

from ivy_models.alexnet import AlexNet, alexnet


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_alexnet_tiny_img_classification(device, f, fw, batch_shape, load_weights):
    """Test AlexNet image classification."""
    num_classes = 1000
    # load image
    this_dir = os.path.dirname(os.path.realpath(__file__))
    img = ivy.asarray(
        np.load(os.path.join(this_dir, "image_alexnet.npy")),
    )
    model = AlexNet(num_classes, 0)

    if load_weights:
        model = alexnet

    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([5, 6, 3, 38, 112])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([6.8951, 6.7858, 6.4805, 6.4796, 6.2703])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)
