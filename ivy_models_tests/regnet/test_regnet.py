import os
import ivy
import random
import numpy as np
import jax

# Enable x64 support in JAX
jax.config.update("jax_enable_x64", True)
from ivy_models_tests import helpers
from ivy_models import (
    regnet_y_400mf,
    regnet_y_800mf,
    regnet_y_1_6gf,
    regnet_y_3_2gf,
    regnet_y_8gf,
    regnet_y_16gf,
    regnet_y_32gf,
    # regnet_y_128gf,
    regnet_x_400mf,
    regnet_x_800mf,
    regnet_x_1_6gf,
    regnet_x_3_2gf,
    regnet_x_8gf,
    regnet_x_16gf,
    regnet_x_32gf,
)


VARIANTS = {
    "y_400mf": regnet_y_400mf,
    "y_800mf": regnet_y_800mf,
    "y_1_6gf": regnet_y_1_6gf,
    "y_3_2gf": regnet_y_3_2gf,
    "y_8gf": regnet_y_8gf,
    "y_16gf": regnet_y_16gf,
    "y_32gf": regnet_y_32gf,
    # "y_128gf": regnet_y_128gf,
    "x_400mf": regnet_x_400mf,
    "x_800mf": regnet_x_800mf,
    "x_1_6gf": regnet_x_1_6gf,
    "x_3_2gf400": regnet_x_3_2gf,
    "x_8gf": regnet_x_8gf,
    "x_16gf": regnet_x_16gf,
    "x_32gf": regnet_x_32gf,
}

LOGITS = {
    "y_400mf": np.array(),
    "y_800mf": np.array(),
    "y_1_6gf": np.array(),
    "y_3_2gf": np.array(),
    "y_8gf": np.array(),
    "y_16gf": np.array(),
    "y_32gf": np.array(),
    "y_128gf": np.array(),
    "x_400mf": np.array(),
    "x_800mf": np.array(),
    "x_1_6gf": np.array(),
    "x_3_2gf400": np.array(),
    "x_8gf": np.array(),
    "x_16gf": np.array(),
    "x_32gf": np.array(),
    "r152": np.array(),
    # "r152": np.array([0.8051, 0.0473, 0.0094]),
}


load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS.keys()))
model = VARIANTS[model_var](pretrained=load_weights)
v = ivy.to_numpy(model.v)


def test_regnet_img_classification(device, fw):
    """Test RegNet-all_variant image classification."""
    num_classes = 1000
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        helpers.load_and_preprocess_img(
            os.path.join(this_dir, "..", "..", "images", "cat.jpg"),
            256,
            224,
            data_format="NHWC",
            to_ivy=True,
        ),
    )

    model.v = ivy.asarray(v)
    output = model(img)

    # # Cardinality test
    # assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # # Value test
    # if load_weights:
    #     output = output[0]
    #     true_indices = ivy.array([282, 281, 285])
    #     calc_indices = ivy.argsort(output, descending=True)[:3]

    #     assert np.array_equal(true_indices, calc_indices)

    #     true_logits = LOGITS[model_var]
    #     calc_logits = np.take(
    #         helpers.np_softmax(ivy.to_numpy(output)), ivy.to_numpy(calc_indices)
    #     )

    #     assert np.allclose(true_logits, calc_logits, rtol=0.005)
