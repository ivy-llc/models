# global
import os
import ivy
import pytest
import numpy as np

# local
from ivy_models.transformers.helpers import FeedForward, PreNorm
from ivy_models.transformers.perceiver_io import (
    PerceiverIOSpec,
    perceiver_io_img_classification,
)
from ivy_models.transformers.perceiver_io import (
    PerceiverIOSpec,
    perceiver_io_img_classification,
)


# Helpers #
# --------#


def test_feedforward(device, f, fw):
    ivy.seed(seed_value=0)
    feedforward = FeedForward(4, device=device)
    x = ivy.random_uniform(shape=(1, 3, 4), device=device)
    ret = feedforward(x)
    assert list(ret.shape) == [1, 3, 4]


def test_prenorm(device, f, fw):
    ivy.seed(seed_value=0)
    att = ivy.MultiHeadAttention(16, device=device)
    prenorm = PreNorm(16, att, device=device)
    x = ivy.random_uniform(shape=(1, 3, 16), device=device)
    ret = prenorm(x)
    assert list(ret.shape) == [1, 3, 16]


# Perceiver IO #
# -------------#


@pytest.mark.parametrize("load_weights", [True, False])
def test_perceiver_io_img_classification(device, f, fw, load_weights):
def test_perceiver_io_img_classification(device, f, fw, load_weights):
    # params
    input_dim = 3
    num_input_axes = 2
    output_dim = 1000
    batch_shape = [1]
    queries_dim = 1024
    learn_query = True
    batch_shape = [1]
    queries_dim = 1024
    learn_query = True
    network_depth = 8 if load_weights else 1
    num_lat_att_per_layer = 6 if load_weights else 1

    # inputs
    this_dir = os.path.dirname(os.path.realpath(__file__))
    img = ivy.array(
        np.load(os.path.join(this_dir, "img.npy"))[None], dtype="float32", device=device
    )
    queries = (
        None
        if learn_query
        else ivy.random_uniform(shape=batch_shape + [1, queries_dim], device=device)
    )

    spec = PerceiverIOSpec(
        input_dim=input_dim,
        num_input_axes=num_input_axes,
        output_dim=output_dim,
        queries_dim=queries_dim,
        network_depth=network_depth,
        learn_query=learn_query,
        query_shape=[1],
        num_fourier_freq_bands=64,
        num_lat_att_per_layer=num_lat_att_per_layer,
        device=device,
    )
    spec = PerceiverIOSpec(
        input_dim=input_dim,
        num_input_axes=num_input_axes,
        output_dim=output_dim,
        queries_dim=queries_dim,
        network_depth=network_depth,
        learn_query=learn_query,
        query_shape=[1],
        num_fourier_freq_bands=64,
        num_lat_att_per_layer=num_lat_att_per_layer,
        device=device,
    )

    model = perceiver_io_img_classification(spec, pretrained=load_weights)
    model = perceiver_io_img_classification(spec, pretrained=load_weights)

    # output
    output = model(img, queries=queries)

    # cardinality test
    assert output.shape == tuple(batch_shape + [1, output_dim])

    # value test
    if load_weights:
        true_logits = np.array([4.9020147, 4.9349823, 8.04229, 8.167497])
        calc_logits = ivy.to_numpy(output[0, 0])

        def np_softmax(x):
            return np.exp(x) / np.sum(np.exp(x))

        true_indices = np.array([6, 5, 251, 246])
        calc_indices = np.argsort(calc_logits)[-4:]

        assert np.array_equal(true_indices, calc_indices)

        true_probs = np_softmax(true_logits)
        calc_probs = np.take(np_softmax(calc_logits), calc_indices)
        assert np.allclose(true_probs, calc_probs, rtol=0.5)


@pytest.mark.parametrize("learn_query", [True, False])
def test_perceiver_io_flow_prediction(device, f, fw, learn_query):
def test_perceiver_io_flow_prediction(device, f, fw, learn_query):
    # params
    input_dim = 3
    num_input_axes = 3
    output_dim = 2
    batch_shape = [3]
    img_dims = [32, 32]
    queries_dim = 32
    batch_shape = [3]
    img_dims = [32, 32]
    queries_dim = 32

    # inputs
    img = ivy.random_uniform(shape=batch_shape + [2] + img_dims + [3], device=device)
    queries = ivy.random_uniform(shape=batch_shape + img_dims + [32], device=device)

    spec = PerceiverIOSpec(
        input_dim=input_dim,
        num_input_axes=num_input_axes,
        output_dim=output_dim,
        queries_dim=queries_dim,
        network_depth=1,
        learn_query=learn_query,
        query_shape=img_dims,
        max_fourier_freq=img_dims[0],
        num_lat_att_per_layer=1,
        device=device,
    )
    # model call
    model = perceiver_io_img_classification(spec, pretrained=False)
    spec = PerceiverIOSpec(
        input_dim=input_dim,
        num_input_axes=num_input_axes,
        output_dim=output_dim,
        queries_dim=queries_dim,
        network_depth=1,
        learn_query=learn_query,
        query_shape=img_dims,
        max_fourier_freq=img_dims[0],
        num_lat_att_per_layer=1,
        device=device,
    )
    # model call
    model = perceiver_io_img_classification(spec, pretrained=False)

    # output
    output = model(img, queries=queries)

    # cardinality test
    assert output.shape == tuple(batch_shape + img_dims + [output_dim])
