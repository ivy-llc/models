import os
import ivy
import pytest
import numpy as np

from ivy_models.convnext import (
    convnext_tiny,
    convnext_small, 
    convnext_base,
    convnext_large,
)

@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_tiny_img_classification(device, f, fw, batch_shape, load_weights):
    """Test ConvNeXt tiny image classification."""
    num_classes = 1000
    device="cpu"
    # load image
    this_dir = os.path.dirname(os.path.realpath(__file__))
    img = ivy.asarray(
        np.load(os.path.join(this_dir, "image_convnext.npy")),
    )
    model = convnext_tiny()

    if load_weights:
        weight_fpath = os.path.join(
            this_dir, "../../ivy_models/convnext/pretrained_weights/convnext_tiny.pkl"
        )

        assert os.path.isfile(weight_fpath)
        
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            pytest.skip()

        model = convnext_tiny(v)
    
    
    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([  0,   1,  78, 303, 111])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([4.9245, 3.5395, 3.2072, 2.9629, 2.9589])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)

@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_small_img_classification(device, f, fw, batch_shape, load_weights):
    """Test ConvNeXt small image classification."""
    num_classes = 1000
    device="cpu"
    
    # Load image
    img_path = "convnext/image_convnext.npy"
    img = ivy.asarray(np.load(img_path), device=device)

    model = convnext_small()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir, "../../ivy_models/convnext/pretrained_weights/convnext_small.pickled"
        )

        assert os.path.isfile(weight_fpath)
        
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            pytest.skip()

        model = convnext_small(v)

    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    if load_weights:
        # Value test
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([623, 111,  21,   1, 644])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([2.5510, 2.5314, 2.4917, 2.2801, 2.2450])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)

@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_base_img_classification(device, f, fw, batch_shape, load_weights):
    """Test ConvNeXt base image classification."""
    num_classes = 1000
    device="cpu"
    
    # Load image
    img_path = "convnext/image_convnext.npy"
    img = ivy.asarray(np.load(img_path), device=device)
    model = convnext_base()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir, "../../ivy_models/convnext/pretrained_weights/convnext_base.pickled"
        )

        assert os.path.isfile(weight_fpath)
        
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            pytest.skip()

        model = convnext_base(v)

    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([623, 111,  21,   1, 644])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([2.5510, 2.5314, 2.4917, 2.2801, 2.2450])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)

@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_convnext_large_img_classification(device, f, fw, batch_shape, load_weights):
    """Test ConvNeXt large image classification."""
    num_classes = 1000
    device="cpu"

    # Load image
    img_path = "convnext/image_convnext.npy"
    img = ivy.asarray(np.load(img_path), device=device)
    model = convnext_large()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir, "../../ivy_models/convnext/pretrained_weights/convnext_large.pickled"
        )
        assert os.path.isfile(weight_fpath)
        
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            pytest.skip()

        model = convnext_large(v)

    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = np.array([  0, 111,  78, 623, 940])
        calc_indices = np.argsort(np_out)[-5:][::-1]
        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([4.2831, 3.3038, 2.4085, 2.3058, 2.1760])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-3)
