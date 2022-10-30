import os
import ivy
import pytest
import numpy as np

from PIL import Image

from ivy_models.transformers.helpers import FeedForward, PreNorm
from ivy_models.transformers.perceiver_io import PerceiverIOSpec, PerceiverIO

import ivy
ivy.set_backend("torch")


# Perceiver IO #
# -------------#

def perceiver_io_img_classification(device, backend, batch_shape, img_dims, queries_dim, learn_query,
                                         load_weights):
    # params
    input_dim = 3
    num_input_axes = 2
    output_dim = 1000
    network_depth = 8 if load_weights else 1
    num_lat_att_per_layer = 6 if load_weights else 1

    # inputs
    # this_dir = os.path.dirname(os.path.realpath(__file__))
    img_raw = Image.open("n01443537_goldfish.jpeg").resize((224, 224))
    img = np.array(img_raw)
    print(img.shape)
    img = img.astype("float32")
    img /= 255

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])

    img[:, :] -= mean
    img[:, :] /= std

    img = ivy.array( img[None], dtype='float32', device=device)

    # print(img.shape)
    # return

    queries = None if learn_query else ivy.random_uniform(shape=batch_shape + [1, queries_dim], device=device)

    model = PerceiverIO(PerceiverIOSpec(input_dim=input_dim,
                                        num_input_axes=num_input_axes,
                                        output_dim=output_dim,
                                        queries_dim=queries_dim,
                                        network_depth=network_depth,
                                        learn_query=learn_query,
                                        query_shape=[1],
                                        num_fourier_freq_bands=64,
                                        num_lat_att_per_layer=num_lat_att_per_layer,
                                        device=device))
    
    # weight_fpath = 'ivy_models/transformers/pretrained_weights/perceiver_io.pickled'
    # assert os.path.isfile(weight_fpath)
    # # noinspection PyBroadException
    
    # v = ivy.Container.from_disk_as_pickled(weight_fpath).from_numpy().as_variables()
    # # noinspection PyUnboundLocalVariable
    # assert ivy.Container.identical_structure([model.v, v])

    # model = PerceiverIO(PerceiverIOSpec(input_dim=input_dim,
    #                                     num_input_axes=num_input_axes,
    #                                     output_dim=output_dim,
    #                                     queries_dim=queries_dim,
    #                                     network_depth=network_depth,
    #                                     learn_query=learn_query,
    #                                     query_shape=[1],
    #                                     max_fourier_freq=img_dims[0],
    #                                     num_fourier_freq_bands=64,
    #                                     num_lat_att_per_layer=num_lat_att_per_layer,
    #                                     device=device), v=v)

    # output
    output = model(img, queries=queries)
    return output
    print(output.shape)

    # cardinality test
    assert output.shape == tuple(batch_shape + [1, output_dim])

device = "cpu"
batch_shape = [1]
img_dims = [224, 224]
queries_dim = 1024
learn_query = [True]
load_weights = False
backend = "numpy"

logits = perceiver_io_img_classification(device, backend, batch_shape, img_dims, queries_dim, learn_query,
                                         load_weights)

print(ivy.argmax(logits, axis=2))