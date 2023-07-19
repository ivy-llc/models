import os
import ivy
import pytest
import numpy as np
import jax
from hypothesis import given, strategies as st

# Enable x64 support in JAX
jax.config.update("jax_enable_x64", True)
from ivy_models_tests import helpers
from ivy_models.resnet import resnet_18, resnet_34, resnet_50, resnet_152, resnet_101


VARIANTS = {
    "r18": resnet_18,
    "r34": resnet_34,
    "r50": resnet_50,
    "r101": resnet_101,
    "r152": resnet_152,
}

LOGITS = {
    "r18": np.array([0.7069, 0.2663, 0.0231]),
    "r34": np.array([0.8507, 0.1351, 0.0069]),
    "r50": np.array([0.3429, 0.0408, 0.0121]),
    "r101": np.array([0.7834, 0.0229, 0.0112]),
    "r152": np.array([0.8051, 0.0473, 0.0094]),
}


def np_softmax(inputs):
    """apply the softmax on the output"""
    return np.exp(inputs) / np.sum(np.exp(inputs))


@pytest.mark.parametrize("load_weights", [False, True])
@given(model_var=st.sampled_from(list(VARIANTS.keys())))
def test_resnet_img_classification(device, f, fw, load_weights, model_var):
    """Test ResNet-18 image classification."""
    num_classes = 1000
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        helpers.load_and_preprocess_img(
            os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
        ),
    )

    # Create model
    model = VARIANTS[model_var](pretrained=load_weights)

    # Perform inference
    output = model(img)

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        output = output[0]
        true_indices = ivy.array([282, 281, 285])
        calc_indices = ivy.argsort(output, descending=True)[:3]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = LOGITS[model_var]
        calc_logits = np.take(
            np_softmax(ivy.to_numpy(output)), ivy.to_numpy(calc_indices)
        )

        assert np.allclose(true_logits, calc_logits, rtol=0.005)


# @pytest.mark.parametrize("batch_shape", [[1]])
# @pytest.mark.parametrize("load_weights", [False, True])
# def test_resnet_34_img_classification(device, f, fw, batch_shape, load_weights):
#     """Test ResNet-34 image classification."""
#     num_classes = 1000
#     this_dir = os.path.dirname(os.path.realpath(__file__))

#     # Load image
#     img = ivy.asarray(
#         helpers.load_and_preprocess_img(
#             os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
#         )
#     )

#     # Create model
#     model = resnet_34(pretrained=load_weights)

#     # Perform inference
#     output = model(img)

#     # Cardinality test
#     assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

#     # Value test
#     if load_weights:
#         output = output[0]
#         true_indices = ivy.array([282, 281, 285])
#         calc_indices = ivy.argsort(output, descending=True)[:3]

#         assert np.array_equal(true_indices, calc_indices)

#         true_logits = np.array([0.8507, 0.1351, 0.0069])
#         calc_logits = np.take(
#             np_softmax(ivy.to_numpy(output)), ivy.to_numpy(calc_indices)
#         )

#         assert np.allclose(true_logits, calc_logits, rtol=0.005)


# @pytest.mark.parametrize("batch_shape", [[1]])
# @pytest.mark.parametrize("load_weights", [False, True])
# def test_resnet_50_img_classification(device, f, fw, batch_shape, load_weights):
#     """Test ResNet-50 image classification."""
#     num_classes = 1000
#     this_dir = os.path.dirname(os.path.realpath(__file__))

#     # Load image
#     img = ivy.asarray(
#         helpers.load_and_preprocess_img(
#             os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
#         )
#     )

#     # Create model
#     model = resnet_50(pretrained=load_weights)

#     # Perform inference
#     output = model(img)

#     # Cardinality test
#     assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

#     # Value test
#     if load_weights:
#         output = output[0]
#         true_indices = ivy.array([282, 281, 285])
#         calc_indices = ivy.argsort(output, descending=True)[:3]

#         assert np.array_equal(true_indices, calc_indices)

#         true_logits = np.array([0.3429, 0.0408, 0.0121])
#         calc_logits = np.take(
#             np_softmax(ivy.to_numpy(output)), ivy.to_numpy(calc_indices)
#         )

#         assert np.allclose(true_logits, calc_logits, rtol=0.005)


# @pytest.mark.parametrize("batch_shape", [[1]])
# @pytest.mark.parametrize("load_weights", [False, True])
# def test_resnet_101_img_classification(device, f, fw, batch_shape, load_weights):
#     """Test ResNet-101 image classification."""
#     num_classes = 1000
#     this_dir = os.path.dirname(os.path.realpath(__file__))

#     # Load image
#     img = ivy.asarray(
#         helpers.load_and_preprocess_img(
#             os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
#         )
#     )

#     # Create model
#     model = resnet_101(pretrained=load_weights)

#     # Perform inference
#     output = model(img)

#     # Cardinality test
#     assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

#     # Value test
#     if load_weights:
#         output = output[0]
#         true_indices = ivy.array([282, 281, 285])
#         calc_indices = ivy.argsort(output, descending=True)[:3]

#         assert np.array_equal(true_indices, calc_indices)

#         true_logits = np.array([0.7834, 0.0229, 0.0112])
#         calc_logits = np.take(
#             np_softmax(ivy.to_numpy(output)), ivy.to_numpy(calc_indices)
#         )

#         assert np.allclose(true_logits, calc_logits, rtol=0.005)


# @pytest.mark.parametrize("batch_shape", [[1]])
# @pytest.mark.parametrize("load_weights", [False, True])
# def test_resnet_152_img_classification(device, f, fw, batch_shape, load_weights):
#     """Test ResNet-152 image classification."""
#     num_classes = 1000
#     this_dir = os.path.dirname(os.path.realpath(__file__))

#     # Load image
#     img = ivy.asarray(
#         helpers.load_and_preprocess_img(
#             os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
#         )
#     )

#     # Create model
#     model = resnet_152(pretrained=load_weights)

#     # Perform inference
#     output = model(img)

#     # Cardinality test
#     assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

#     # Value test
#     if load_weights:
#         output = output[0]
#         true_indices = ivy.array([282, 281, 285])
#         calc_indices = ivy.argsort(output, descending=True)[:3]

#         assert np.array_equal(true_indices, calc_indices)

#         true_logits = np.array([0.8051, 0.0473, 0.0094])
#         calc_logits = np.take(
#             np_softmax(ivy.to_numpy(output)), ivy.to_numpy(calc_indices)
#         )

#         assert np.allclose(true_logits, calc_logits, rtol=0.005)
