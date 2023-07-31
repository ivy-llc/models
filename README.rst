.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/logo.png?raw=true#gh-light-mode-only
   :width: 100%
   :class: only-light

.. image:: https://github.com/unifyai/unifyai.github.io/blob/master/img/externally_linked/logo_dark.png?raw=true#gh-dark-mode-only
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

Ivy Models
===========

This repository houses a collection of popular machine learning models written in `Ivy <https://github.com/unifyai/ivy>`_.

Models written in Ivy are compatible with PyTorch, TensorFlow, JAX and NumPy.
This means that these models can be used as part of a working pipeline in any of the standard ML frameworks.
The purpose of this repository is to provide reference Ivy implementations of common machine learning models.
It also gives a demonstration of how to write custom models in Ivy.

Check out our `demos <https://unify.ai/demos/#examples-and-demos>`_ to see these models in action.
In particular, `UNet <https://unify.ai/demos/examples_and_demos/image_segmentation_with_ivy_unet.html>`_ 
and `AlexNet <https://unify.ai/demos/examples_and_demos/alexnet_demo.html>`_ demonstrate using models from this repository.

These models can be loaded with pretrained weights, we have tests to ensure that our models give the same output as the reference implementation.
Models can also be initialised with random weights by passing :code:`pretrained=False` to the loading function.

To learn more about Ivy, check out `unify.ai <https://unify.ai>`_, our `Docs <https://unify.ai/docs/ivy/>`_, and our `GitHub <https://github.com/unifyai/ivy>`_.

Setting up
------------
.. code-block:: bash

    git clone https://github.com/unifyai/models
    cd models
    pip install .

Getting started
-----------------

.. code-block:: python

    import ivy
    from ivy_models import alexnet
    ivy.set_backend(“torch”)
    model = alexnet()

The AlexNet model is now ready to be used, and is compatible with any other PyTorch code.
See `this demo <https://unify.ai/demos/examples_and_demos/alexnet_demo.html>`_ for more details.

Navigating this repository
-----------------------------
The models are contained in the ivy_models folder.
The functions that automatically load the pretrained weights are found at the end of :code:`model_name.py`, some models have multiple sizes.
The layers are sometimes kept in a separate file, usually named :code:`layers.py`.


Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
