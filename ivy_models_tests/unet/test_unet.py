# import os
# import ivy
# import pytest
# import numpy as np
# import jax

# # Enable x64 support in JAX
# jax.config.update("jax_enable_x64", True)

# from ivy_models.unet import UNet
# from ivy_models_tests import helpers


# @pytest.mark.parametrize("batch_shape", [[1]])
# @pytest.mark.parametrize("load_weights", [False, True])
# def test_unet_img_segmentation(device, f, fw, batch_shape, load_weights):
#     """Test ResNet-18 image classification."""
#     num_classes = 1000
#     this_dir = os.path.dirname(os.path.realpath(__file__))

#     # Load image
#     img = helpers.load_and_preprocess_img(
#         os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
#     )

#     # Create model
#     model = UNet(pretrained=load_weights)

#     # Perform inference
#     output = model(img)

#     # Cardinality test
#     assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])
