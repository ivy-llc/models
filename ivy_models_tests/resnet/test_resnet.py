import os
import ivy
import pytest
import numpy as np
import jax

# Enable x64 support in JAX
jax.config.update("jax_enable_x64", True)

from ivy_models.resnet import resnet_18


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_resnet_18_img_classification(device, f, fw, batch_shape, load_weights):
    """Test ResNet-18 image classification."""
    num_classes = 1000
    device = "cpu"
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = ivy.asarray(
        np.load(os.path.join(this_dir, "img_resnet.npy"))[None],
        dtype="float32",
        device=device,
    )

    # Create model
    model = resnet_18()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir,
            "../../ivy_models/resnet/pretrained_weights/resnet_18.pickled",
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

        model = resnet_18(v=v)

    # Perform inference
    output = model(img[0])

    if load_weights:
        if isinstance(output, ivy.Container):
            output = output["w"]

    # Cardinality test
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        np_out = ivy.to_numpy(output[0])
        true_indices = np.array([412, 731, 600, 463])
        calc_indices = np.argsort(np_out)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([2.0218, 2.1013, 2.1893, 2.4626])
        calc_logits = np.take(np_out, calc_indices)

        assert np.allclose(true_logits, calc_logits, rtol=0.5)
