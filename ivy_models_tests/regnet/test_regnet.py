from ivy_models.regnet import regnet_y_400mf, regnet_y_800mf
from ivy_models_tests import helpers
import ivy
import random
import os


VARIANTS = {
    "regnet_y_400mf": regnet_y_400mf,
    "regnet_y_800mf": regnet_y_800mf,
}

load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS.keys()))
model = VARIANTS[model_var](pretrained=load_weights)
v = ivy.to_numpy(model.v)


def test_regnet(device, fw):
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
        )
    )

    # Create model
    model.v = ivy.asarray(v)
    logits = model(img)

    # Cardinality test
    assert logits.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(logits[0])
        true_indices = ivy.to_numpy(ivy.sort(ivy.array([282, 281, 285, 287])))
        calc_indices = ivy.to_numpy(ivy.sort(ivy.argsort(np_out)[-5:][::-1]))
        assert ivy.array_equal(true_indices, calc_indices[:4])
