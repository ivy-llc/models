import os
import ivy
import pytest
import numpy as np

from ivy_models.mlpmixer import mlpmixer
from ivy_models_tests import helpers

import tensorflow as tf
from tensorflow import keras
from keras import layers
import jax

jax.config.update("jax_enable_x64", False)

load_weights = True
model = mlpmixer(pretrained=load_weights)
v = ivy.to_numpy(model.v)


@pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
def test_mlpmixer_tiny_img_classification(device, fw, data_format):
    """Test MLPMixer image classification."""
    num_classes = 10
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_image_in_np(
        os.path.join(this_dir, "..", "..", "images", "car.jpg")
    )

    # Preprocess the image
    def get_augmentation_layers():
        data_augmentation = keras.Sequential(
            [
                layers.experimental.preprocessing.Normalization(
                    mean=(0.5, 0.5, 0.5), variance=(0.25, 0.25, 0.25)
                ),
                layers.experimental.preprocessing.Resizing(72, 72),
                layers.experimental.preprocessing.RandomFlip("horizontal"),
                layers.experimental.preprocessing.RandomRotation(factor=0.02),
                layers.experimental.preprocessing.RandomZoom(
                    height_factor=0.2, width_factor=0.2
                ),
            ],
            name="data_augmentation",
        )
        return data_augmentation

    data_augmentation = get_augmentation_layers()
    img = data_augmentation(img)
    img = tf.expand_dims(img, 0).numpy()
    img = ivy.asarray(img)
    if data_format == "NCHW":
        img = ivy.permute_dims(img, (0, 3, 1, 2))

    model.v = ivy.asarray(v)
    logits = model(img, data_format=data_format)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits)
        true_indices = np.array([4, 7, 2, 9])
        calc_indices = np.argsort(np_out[0])[-4:][::-1]
        assert np.array_equal(np.sort(true_indices), np.sort(calc_indices))

        true_logits = np.array([0.4022081, 0.24405026, 0.14345096, 0.12923254])
        calc_logits = np.take(np_out, calc_indices)
        assert np.allclose(true_logits, calc_logits, rtol=1e-2, atol=1e-1)
