.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/logo.png?raw=true#gh-light-mode-only
   :width: 100%
   :class: only-light

.. image:: https://github.com/unifyai/unifyai.github.io/blob/main/img/externally_linked/logo_dark.png?raw=true#gh-dark-mode-only
   :width: 100%
   :class: only-dark


.. raw:: html

    <br/>
    <a href="https://pypi.org/project/ivy-models">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://badge.fury.io/py/ivy-models.svg">
    </a>
    <a href="https://github.com/unifyai/models/actions?query=workflow%3Adocs">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/models/actions/workflows/docs.yml/badge.svg">
    </a>
    <a href="https://github.com/unifyai/models/actions?query=workflow%3Anightly-tests">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://github.com/unifyai/models/actions/workflows/nightly-tests.yml/badge.svg">
    </a>
    <a href="https://discord.gg/G4aR9Q7DTN">
        <img class="dark-light" style="float: left; padding-right: 4px; padding-bottom: 4px;" src="https://img.shields.io/discord/799879767196958751?color=blue&label=%20&logo=discord&logoColor=white">
    </a>
    <br clear="all" />

Perceiver IO
===========

`Perceiver IO <https://arxiv.org/abs/2107.14795>`_  is based on the Perceiver architecture, which was originally proposed by Google AI in 2021. Perceiver IO extends the Perceiver architecture 
by adding a new module called the Querying Module. The Querying Module allows Perceiver IO to produce outputs of arbitrary size and semantics, 
which makes it a more general-purpose architecture than the Perceiver.

The Perceiver IO architecture consists of three main modules. These are the reading module which takes the input data and encodes it into a latent space, 
the processing module which refines the latent representation learned by the reading module and the querying module which takes the latent 
representation from the Processing Module and produces outputs of arbitrary size and semantics.

The Querying Module is the key innovation of Perceiver IO. It works by first constructing a query vector for each output element. 
The query vector is a representation of the desired output element, and it is constructed using the output-specific features. 
The Querying Module then uses a self-attention mechanism to attend to the latent representation, and it produces the output element by combining 
the latent representation with the query vector.

Getting started
-----------------

.. code-block:: python

    import ivy
    from ivy_models.transformers.perceiver_io import (
        PerceiverIOSpec,
        perceiver_io_img_classification,
    )
    ivy.set_backend("torch")

    # params
    input_dim = 3
    num_input_axes = 2
    output_dim = 1000
    batch_shape = [1]
    queries_dim = 1024
    learn_query = True
    network_depth = 8 if load_weights else 1
    num_lat_att_per_layer = 6 if load_weights else 1

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
        device='cuda',
    )

    model = perceiver_io_img_classification(spec)

The pretrained perceiver_io_img_classification model is now ready to be used!!!

Citation
--------

::

    @article{
      title={Perceiver IO: A General Architecture for Structured Inputs & Outputs},
      author={
        Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, 
        Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, Olivier Hénaff, Matthew M. Botvinick, 
        Andrew Zisserman, Oriol Vinyals and Joāo Carreira
      },
      journal={arXiv preprint arXiv:2107.14795},
      year={2022}
    }


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
