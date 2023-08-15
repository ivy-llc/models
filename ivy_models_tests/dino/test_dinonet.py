import os
import ivy
import numpy as np
# import pytest
import traceback
import sys
import logging
from ivy_models_tests import helpers
from ivy_models.dino.dino import dino_base


# @pytest.mark.parametrize("data_format", ["NHWC", "NCHW"])
# def test_dino_classification(device, fw, data_format):
#     """Test AlexNet image classification."""
#     num_classes = 1000
#     batch_shape = [1]
#     this_dir = os.path.dirname(os.path.realpath(__file__))
#
#     # Load image
#     img = helpers.load_and_preprocess_img(
#         os.path.join(this_dir, "..", "..", "images", "cat.jpg"),
#         256,
#         224,
#         data_format=data_format,
#         to_ivy=True,
#     )
#
#     model = dino_base()
#
    
def run_model():
    num_classes = 1000
    batch_shape = [1]
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"),
        256,
        224,
        data_format="NHWC",
        to_ivy=True,
    )

    model = dino_base()

    try:
        print(type(img))
        print(model.v)
        model.v = ivy.asarray(model.v)
        logits = model(img)
        print("LOGITS")
        print(logits)
    except Exception as e:
        print(traceback.format_exc())
        # or
        print(sys.exc_info()[2])

run_model()
