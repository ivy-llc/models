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
    # regnet_y_800mf,
    # regnet_y_1_6gf,
    # regnet_y_3_2gf,
    # regnet_y_8gf,
    # regnet_y_16gf,
    # regnet_y_32gf,
    # # regnet_y_128gf,
    # regnet_x_400mf,
    # regnet_x_800mf,
    # regnet_x_1_6gf,
    # regnet_x_3_2gf,
    # regnet_x_8gf,
    # regnet_x_16gf,
    # regnet_x_32gf,
)


VARIANTS = {
    "y_400mf": regnet_y_400mf,
    # "y_800mf": regnet_y_800mf,
    # "y_1_6gf": regnet_y_1_6gf,
    # "y_3_2gf": regnet_y_3_2gf,
    # "y_8gf": regnet_y_8gf,
    # "y_16gf": regnet_y_16gf,
    # "y_32gf": regnet_y_32gf,
    # # "y_128gf": regnet_y_128gf,
    # "x_400mf": regnet_x_400mf,
    # "x_800mf": regnet_x_800mf,
    # "x_1_6gf": regnet_x_1_6gf,
    # "x_3_2gf400": regnet_x_3_2gf,
    # "x_8gf": regnet_x_8gf,
    # "x_16gf": regnet_x_16gf,
    # "x_32gf": regnet_x_32gf,
}

LOGITS = {
    "y_400mf": np.array([0.7257, 0.2450, 0.0275]),
    # "y_800mf": np.array([0.6676, 0.2091, 0.1194]),
    # "y_1_6gf": np.array([0.7617, 0.2035, 0.0319]),
    # "y_3_2gf": np.array([0.8119, 0.1718, 0.0143]),
    # "y_8gf": np.array([0.9308, 0.0630, 0.0059]),
    # "y_16gf": np.array([0.7894, 0.2074, 0.0024]),
    # "y_32gf": np.array([0.8221, 0.1764, 0.0014]),
    # "x_400mf": np.array([0.6707, 0.2706, 0.0517]),
    # "x_800mf": np.array([0.5628, 0.3851, 0.0453]),
    # "x_1_6gf": np.array([0.6949, 0.2738, 0.0293]),
    # "x_3_2gf400": np.array([0.8212, 0.1721, 0.0055]),
    # "x_8gf": np.array([0.8271, 0.1659, 0.0063]),
    # "x_16gf": np.array([0.7613, 0.2340, 0.0043]),
    # "x_32gf": np.array([0.7828, 0.2146, 0.0022]),
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
