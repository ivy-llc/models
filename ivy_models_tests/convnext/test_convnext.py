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
    img_path = "convnext/image_convnext.npy"
    img = ivy.asarray(np.load(img_path), device=device)

    # load weights
    this_dir = os.path.dirname(os.path.realpath(__file__))
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
    # load image
    img_path = "convnext/image_convnext.npy"
    img = ivy.asarray(np.load(img_path), device=device)

    # load weights
    this_dir = os.path.dirname(os.path.realpath(__file__))
    weight_fpath = os.path.join(
        this_dir, "../../ivy_models/convnext/pretrained_weights/convnext_small.pkl"
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
    # load image
    img_path = "convnext/image_convnext.npy"
    img = ivy.asarray(np.load(img_path), device=device)

    # load weights
    this_dir = os.path.dirname(os.path.realpath(__file__))
    weight_fpath = os.path.join(
        this_dir, "../../ivy_models/convnext/pretrained_weights/convnext_base.pkl"
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
    # load image
    img_path = "convnext/image_convnext.npy"
    img = ivy.asarray(np.load(img_path), device=device)

    # load weights
    this_dir = os.path.dirname(os.path.realpath(__file__))
    weight_fpath = os.path.join(
        this_dir, "../../ivy_models/convnext/pretrained_weights/convnext_large.pkl"
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
    ## TODO: compute the true logits & indices
    np_out = ivy.to_numpy(logits[0])
    #true_indices = np.array([111,   1, 644,   0, 318])
    calc_indices = np.argsort(np_out)[-5:][::-1]
    assert np.array_equal(true_indices, calc_indices)

    #true_logits = np.array([2.9980, 2.4199, 2.2514, 2.1669, 2.1209]])
    calc_logits = np.take(np_out, calc_indices)
    assert np.allclose(true_logits, calc_logits, rtol=1e-3)


for backend in ["torch"]: #['numpy', 'jax', 'tensorflow', 'torch']:
    ivy.set_backend(backend)
    test_convnext_tiny_img_classification()
    print(f'{backend} passed!')

