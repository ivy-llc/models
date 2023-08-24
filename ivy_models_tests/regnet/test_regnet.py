from ivy_models.regnet import regnet_y_1_6gf
from ivy_models_tests import helpers
import ivy
import random
import os

ivy.set_backend("torch")

VARIANTS = {
    "regnet_y_1_6gf": regnet_y_1_6gf,
}

load_weights = random.choice([False, True])
model_var = random.choice(list(VARIANTS.keys()))
model = VARIANTS[model_var](pretrained=load_weights)
v = ivy.to_numpy(model.v)


def test_regnet_img_classification(device, fw):
    """Test RegNet image classification."""
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
        output = logits[0]
        true_indices = ivy.sort(ivy.array([282, 281, 285, 287]))
        calc_indices = ivy.sort(ivy.argsort(output)[-5:][::-1])

        assert ivy.array_equal(true_indices, calc_indices[:4])
