import os
import pytest
import numpy as np
import jax
import ivy

from ivy_models.vgg import (
    vgg11,
    vgg11_bn,
    vgg13,
    vgg13_bn,
    vgg16,
    vgg16_bn,
    vgg19,
    vgg19_bn,
)


# Enable x64 support in JAX
jax.config.update("jax_enable_x64", True)


def normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """prepare image to be the model's input"""
    img -= np.array(mean)
    img /= np.array(std)
    return img


def np_softmax(inputs):
    """apply the softmax on the output"""
    return np.exp(inputs) / np.sum(np.exp(inputs))


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_vgg_11_img_classification(device, f, fw, batch_shape, load_weights):
    """Test VGG-11 image classification."""
    num_classes = 1000
    device = "cpu"
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        normalize(np.load(os.path.join(this_dir, "img_classification_test.npy"))[None]),
        dtype="float32",
        device=device,
    )

    # Create model
    model = vgg11()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir,
            "../ivy_models/vgg/pretrained_weights/vgg11.pickled",
        )

        # Check if weight file exists
        assert os.path.isfile(weight_fpath)

        # Load weights
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            # If git large-file-storage is not enabled
            # (for example when testing in github actions workflow), then the
            #  test will fail here. A placeholder file does exist,
            #  but the file cannot be loaded as pickled variables.
            pytest.skip()

        model = vgg11(v=v)

    # Perform inference
    output = model(img[0])

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(output[0])
        true_indices = np.array([287, 285, 281, 282])
        calc_indices = np.argsort(np_out)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([0.0236, 0.0563, 0.3473, 0.5525])
        calc_logits = np.take(np_softmax(np_out), calc_indices)

        assert np.allclose(true_logits, calc_logits, rtol=0.005)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_vgg_11_bn_img_classification(device, f, fw, batch_shape, load_weights):
    """Test VGG-11-BN image classification."""
    num_classes = 1000
    device = "cpu"
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        normalize(np.load(os.path.join(this_dir, "img_classification_test.npy"))[None]),
        dtype="float32",
        device=device,
    )

    # Create model
    model = vgg11_bn()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir,
            "../ivy_models/vgg/pretrained_weights/vgg11_bn.pickled",
        )

        # Check if weight file exists
        assert os.path.isfile(weight_fpath)

        # Load weights
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            # If git large-file-storage is not enabled
            # (for example when testing in github actions workflow), then the
            #  test will fail here. A placeholder file does exist,
            #  but the file cannot be loaded as pickled variables.
            pytest.skip()

        model = vgg11_bn(v=v)

    # Perform inference
    output = model(img[0])

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(output[0])
        true_indices = np.array([287, 285, 281, 282])
        calc_indices = np.argsort(np_out)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([0.00136, 0.0843, 0.2417, 0.6685])
        calc_logits = np.take(np_softmax(np_out), calc_indices)

        assert np.allclose(true_logits, calc_logits, rtol=0.005)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_vgg_13_img_classification(device, f, fw, batch_shape, load_weights):
    """Test VGG-13 image classification."""
    num_classes = 1000
    device = "cpu"
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        normalize(np.load(os.path.join(this_dir, "img_classification_test.npy"))[None]),
        dtype="float32",
        device=device,
    )

    # Create model
    model = vgg13()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir,
            "../ivy_models/vgg/pretrained_weights/vgg13.pickled",
        )

        # Check if weight file exists
        assert os.path.isfile(weight_fpath)

        # Load weights
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            # If git large-file-storage is not enabled
            # (for example when testing in github actions workflow), then the
            #  test will fail here. A placeholder file does exist,
            #  but the file cannot be loaded as pickled variables.
            pytest.skip()

        model = vgg13(v=v)

    # Perform inference
    output = model(img[0])

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(output[0])
        true_indices = np.array([292, 285, 281, 282])
        calc_indices = np.argsort(np_out)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([0.00802, 0.01459, 0.14998, 0.8200])
        calc_logits = np.take(np_softmax(np_out), calc_indices)

        assert np.allclose(true_logits, calc_logits, rtol=0.005)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_vgg_13_bn_img_classification(device, f, fw, batch_shape, load_weights):
    """Test VGG-13-BN image classification."""
    num_classes = 1000
    device = "cpu"
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        normalize(np.load(os.path.join(this_dir, "img_classification_test.npy"))[None]),
        dtype="float32",
        device=device,
    )

    # Create model
    model = vgg13_bn()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir,
            "../ivy_models/vgg/pretrained_weights/vgg13_bn.pickled",
        )

        # Check if weight file exists
        assert os.path.isfile(weight_fpath)

        # Load weights
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            # If git large-file-storage is not enabled
            # (for example when testing in github actions workflow), then the
            #  test will fail here. A placeholder file does exist,
            #  but the file cannot be loaded as pickled variables.
            pytest.skip()

        model = vgg13_bn(v=v)

    # Perform inference
    output = model(img[0])

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(output[0])
        true_indices = np.array([292, 285, 281, 282])
        calc_indices = np.argsort(np_out)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([0.00565121, 0.01147511, 0.17541069, 0.8058933])
        calc_logits = np.take(np_softmax(np_out), calc_indices)

        assert np.allclose(true_logits, calc_logits, rtol=0.005)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_vgg_16_img_classification(device, f, fw, batch_shape, load_weights):
    """Test VGG-16 image classification."""
    num_classes = 1000
    device = "cpu"
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        normalize(np.load(os.path.join(this_dir, "img_classification_test.npy"))[None]),
        dtype="float32",
        device=device,
    )

    # Create model
    model = vgg16()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir,
            "../ivy_models/vgg/pretrained_weights/vgg16.pickled",
        )

        # Check if weight file exists
        assert os.path.isfile(weight_fpath)

        # Load weights
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            # If git large-file-storage is not enabled
            # (for example when testing in github actions workflow), then the
            #  test will fail here. A placeholder file does exist,
            #  but the file cannot be loaded as pickled variables.
            pytest.skip()

        model = vgg16(v=v)

    # Perform inference
    output = model(img[0])

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(output[0])
        true_indices = np.array([24, 285, 281, 282])
        calc_indices = np.argsort(np_out)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([0.00315, 0.02750, 0.34825, 0.61424])
        calc_logits = np.take(np_softmax(np_out), calc_indices)

        assert np.allclose(true_logits, calc_logits, rtol=0.005)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_vgg_16_bn_img_classification(device, f, fw, batch_shape, load_weights):
    """Test VGG-16-BN image classification."""
    num_classes = 1000
    device = "cpu"
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        normalize(np.load(os.path.join(this_dir, "img_classification_test.npy"))[None]),
        dtype="float32",
        device=device,
    )

    # Create model
    model = vgg16_bn()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir,
            "../ivy_models/vgg/pretrained_weights/vgg16_bn.pickled",
        )

        # Check if weight file exists
        assert os.path.isfile(weight_fpath)

        # Load weights
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            # If git large-file-storage is not enabled
            # (for example when testing in github actions workflow), then the
            #  test will fail here. A placeholder file does exist,
            #  but the file cannot be loaded as pickled variables.
            pytest.skip()

        model = vgg16_bn(v=v)

    # Perform inference
    output = model(img[0])

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(output[0])
        true_indices = np.array([622, 285, 281, 282])
        calc_indices = np.argsort(np_out)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([0.00040706596, 0.021315627, 0.3374813, 0.6393761])
        calc_logits = np.take(np_softmax(np_out), calc_indices)

        assert np.allclose(true_logits, calc_logits, rtol=0.005)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_vgg_19_img_classification(device, f, fw, batch_shape, load_weights):
    """Test VGG-19 image classification."""
    num_classes = 1000
    device = "cpu"
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        normalize(np.load(os.path.join(this_dir, "img_classification_test.npy"))[None]),
        dtype="float32",
        device=device,
    )

    # Create model
    model = vgg19()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir,
            "../ivy_models/vgg/pretrained_weights/vgg19.pickled",
        )

        # Check if weight file exists
        assert os.path.isfile(weight_fpath)

        # Load weights
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            # If git large-file-storage is not enabled
            # (for example when testing in github actions workflow), then the
            #  test will fail here. A placeholder file does exist,
            #  but the file cannot be loaded as pickled variables.
            pytest.skip()

        model = vgg19(v=v)

    # Perform inference
    output = model(img[0])

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(output[0])
        true_indices = np.array([24, 285, 281, 282])
        calc_indices = np.argsort(np_out)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([0.00102419, 0.00657986, 0.4198733, 0.5677029])
        calc_logits = np.take(np_softmax(np_out), calc_indices)

        assert np.allclose(true_logits, calc_logits, rtol=0.005)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_vgg_19_bn_img_classification(device, f, fw, batch_shape, load_weights):
    """Test VGG-19-BN image classification."""
    num_classes = 1000
    device = "cpu"
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        normalize(np.load(os.path.join(this_dir, "img_classification_test.npy"))[None]),
        dtype="float32",
        device=device,
    )

    # Create model
    model = vgg19_bn()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir,
            "../ivy_models/vgg/pretrained_weights/vgg19_bn.pickled",
        )

        # Check if weight file exists
        assert os.path.isfile(weight_fpath)

        # Load weights
        try:
            v = ivy.Container.cont_from_disk_as_pickled(weight_fpath)
            v = ivy.asarray(v)
        except Exception:
            # If git large-file-storage is not enabled
            # (for example when testing in github actions workflow), then the
            #  test will fail here. A placeholder file does exist,
            #  but the file cannot be loaded as pickled variables.
            pytest.skip()

        model = vgg19_bn(v=v)

    # Perform inference
    output = model(img[0])

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(output[0])
        true_indices = np.array([806, 285, 281, 282])
        calc_indices = np.argsort(np_out)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([0.00011722738, 0.060040653, 0.46304694, 0.47583136])
        calc_logits = np.take(np_softmax(np_out), calc_indices)

        assert np.allclose(true_logits, calc_logits, rtol=0.005)
