import os
import ivy
import pytest
import numpy as np
from ivy_models_tests import helpers
from ivy_models.convnext import convnext


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_tiny_img_classification(device, f, fw, batch_shape, load_weights):
    """Test ConvNeXt tiny image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )
    img = ivy.permute_dims(img, (0,3,1,2))
    # Create model
    model = convnext("tiny", pretrained=load_weights)

    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 753, 285, 643])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([8.396295 , 7.442236 , 5.5498557, 5.164446 , 3.710167])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_small_img_classification(device, f, fw, batch_shape, load_weights):
    """Test ConvNeXt small image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )
    img = ivy.permute_dims(img, (0,3,1,2))
    
    # Create model
    model = convnext("small", pretrained=load_weights)
    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    if load_weights:
        # Value test
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 716, 753, 743])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([9.195629 , 6.0763097, 4.8471575, 3.8018446, 3.520919])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_base_img_classification(device, f, fw, batch_shape, load_weights):
    """Test ConvNeXt base image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )
    img = ivy.permute_dims(img, (0,3,1,2))
    
    # Create model
    model = convnext("base", pretrained=load_weights)
    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 292, 743])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([9.925498, 6.706859, 4.071909, 3.399646, 2.406484])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_large_img_classification(device, f, fw, batch_shape, load_weights):
    """Test ConvNeXt large image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))
    
    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )
    img = ivy.permute_dims(img, (0,3,1,2))
    
    # Create model
    model = convnext("large", pretrained=load_weights)

    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([282, 281, 285, 292, 287])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([9.526691 , 6.26409  , 4.3038955, 3.3073874, 1.7843213])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)
