import os
import ivy
import pytest
import numpy as np
from ivy_models_tests import helpers
from ivy_models.convnext import convnext, convnextv2


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_tiny_img_classification(device, fw, batch_shape, load_weights):
    """Test ConvNeXt tiny image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )
    img = ivy.permute_dims(img, (0, 3, 1, 2))
    # Create model
    model = convnext("tiny", pretrained=load_weights)

    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 287, 292])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([8.791083, 6.803193, 5.147233, 2.5118146, 1.3056283])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_small_img_classification(device, fw, batch_shape, load_weights):
    """Test ConvNeXt small image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )
    img = ivy.permute_dims(img, (0, 3, 1, 2))

    # Create model
    model = convnext("small", pretrained=load_weights)
    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    if load_weights:
        # Value test
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 287, 292])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([8.467648, 8.057183, 6.881177, 2.6506257, 1.8245339])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_base_img_classification(device, fw, batch_shape, load_weights):
    """Test ConvNeXt base image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )
    img = ivy.permute_dims(img, (0, 3, 1, 2))

    # Create model
    model = convnext("base", pretrained=load_weights)
    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 287, 292])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([8.707129, 7.919885, 5.577528, 2.378178, 2.0281594])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_large_img_classification(device, fw, batch_shape, load_weights):
    """Test ConvNeXt large image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )
    img = ivy.permute_dims(img, (0, 3, 1, 2))

    # Create model
    model = convnext("large", pretrained=load_weights)

    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 287, 292])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([8.77628, 7.687718, 6.09846, 2.25323, 2.2160888])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnextv2_atto_img_classification(device, fw, batch_shape, load_weights):
    """Test ConvNeXtV2 atto image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )
    img = ivy.permute_dims(img, (0, 3, 1, 2))

    # Create model
    model = convnextv2("atto", pretrained=load_weights)

    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 287, 292])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([7.1058, 6.6685, 5.9932, 2.6573, 2.2070])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, atol=1e-3)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnextv2_base_img_classification(device, fw, batch_shape, load_weights):
    """Test ConvNeXtV2 base image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )
    img = ivy.permute_dims(img, (0, 3, 1, 2))

    # Create model
    model = convnextv2("base", pretrained=load_weights)

    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 287, 292])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([8.5643, 7.6972, 5.9340, 2.7507, 2.3775])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, atol=1e-3)
