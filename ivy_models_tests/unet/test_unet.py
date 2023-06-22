import os
import ivy
import pytest
import numpy as np
from ivy_models.unet import UNet
import jax

# Enable x64 support in JAX
jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_image_img_segmentation(device, f, fw, batch_shape, load_weights):
    """Test UNet image segmentation."""
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
    model = UNet()

    if load_weights:
        this_dir = os.path.dirname(os.path.realpath(__file__))
        weight_fpath = os.path.join(
            this_dir,
            "../../ivy_models/unet/unet_weights.pkl",
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

        model = UNet(v=v)

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
        true_indices = np.array([287, 285, 281, 282])
        calc_indices = np.argsort(np_out)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_logits = np.array([10.133721, 11.933505, 12.343024, 13.348762])
        calc_logits = np.take(np_out, calc_indices)

        assert np.allclose(true_logits, calc_logits, rtol=0.5)
