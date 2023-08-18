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

Ivy Models
===========

This repository houses a collection of popular machine learning models written in the `Ivy framework <https://github.com/unifyai/ivy>`_.

Code written in Ivy is compatible with PyTorch, TensorFlow, JAX and NumPy.
This means that these models can be integrated into a working pipeline for any of these standard ML frameworks.

The purpose of this repository is to provide reference Ivy implementations of common machine learning models, as well as giving a demonstration of how to write custom models in Ivy.

Check out our `demos <https://unify.ai/demos/#examples-and-demos>`_ to see these models in action.
In particular, `UNet <https://unify.ai/demos/examples_and_demos/image_segmentation_with_ivy_unet.html>`_ 
and `AlexNet <https://unify.ai/demos/examples_and_demos/alexnet_demo.html>`_ demonstrate using models from this repository.

The models can be loaded with pretrained weights, we have tests to ensure that our models give the same output as the reference implementation.
Models can also be initialised with random weights by passing :code:`pretrained=False` to the loading function.

To learn more about Ivy, check out `unify.ai <https://unify.ai>`_, our `Docs <https://unify.ai/docs/ivy/>`_, and our `GitHub <https://github.com/unifyai/ivy>`_.

Setting up
------------

.. code-block:: bash

    git clone https://github.com/unifyai/models
    cd models
    pip install .
    pip install -r requirements.txt  # this is not redundant, it installs latest ivy code which is a dependency 😄

Getting started
-----------------

.. code-block:: python

    import ivy
    from ivy_models import alexnet
    ivy.set_backend(“torch”)
    model = alexnet()

The pretrained AlexNet model is now ready to be used, and is compatible with any other PyTorch code.
See `this demo <https://unify.ai/demos/examples_and_demos/alexnet_demo.html>`_ for more details.

Navigating this repository
-----------------------------
The models are contained in the ivy_models folder.
The functions that automatically load the pretrained weights are found at the end of :code:`model_name.py`, some models have multiple sizes.
The layers are sometimes kept in a separate file, usually named :code:`layers.py`.


**Off-the-shelf models for a variety of domains.**

.. raw:: html

    <div style="display: block;" align="center">
        <img class="dark-light" width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://jax.readthedocs.io">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/jax_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://www.tensorflow.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/tensorflow_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://pytorch.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/pytorch_logo.png">
        </a>
        <img class="dark-light" width="12%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
        <a href="https://numpy.org">
            <img class="dark-light" width="13%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/numpy_logo.png">
        </a>
        <img class="dark-light" width="6%" style="float: left;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/supported/empty.png">
    </div>
    <br clear="all" />


**Ivy Libraries**

There are a host of derived libraries written in Ivy, in the areas of mechanics, 3D vision, robotics, gym environments,
neural memory, pre-trained models + implementations, and builder tools with trainers, data loaders and more. Click on the icons below to learn more!

.. raw:: html

    <div style="display: block;">
        <a href="https://github.com/unifyai/mech">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_mech_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_mech.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/vision">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_vision_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_vision.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/robot">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_robot_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_robot.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/gym">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_gym_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_gym.png">
            </picture>
        </a>

        <br clear="all" />

        <a href="https://pypi.org/project/ivy-mech">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-mech.svg">
        </a>
        <a href="https://pypi.org/project/ivy-vision">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-vision.svg">
        </a>
        <a href="https://pypi.org/project/ivy-robot">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-robot.svg">
        </a>
        <a href="https://pypi.org/project/ivy-gym">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;"width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-gym.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/unifyai/mech/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;"src="https://github.com/unifyai/mech/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/vision/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/vision/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/robot/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/robot/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/gym/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/gym/actions/workflows/nightly-tests.yml/badge.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/unifyai/memory">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_memory_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_memory.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/builder">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_builder_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_builder.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/models">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_models_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_models.png">
            </picture>
        </a>
        <a href="https://github.com/unifyai/ecosystem">
            <picture>
                <source class="dark-light" width="15%" style="float: left; margin: 0% 5%;" media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_ecosystem_dark.png">
                <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://raw.githubusercontent.com/unifyai/unifyai.github.io/main/img/externally_linked/logos/ivy_ecosystem.png">
            </picture>
        </a>

        <br clear="all" />

        <a href="https://pypi.org/project/ivy-memory">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-memory.svg">
        </a>
        <a href="https://pypi.org/project/ivy-builder">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-builder.svg">
        </a>
        <a href="https://pypi.org/project/ivy-models">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://badge.fury.io/py/ivy-models.svg">
        </a>
        <a href="https://github.com/unifyai/ecosystem/actions?query=workflow%3Adocs">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/ecosystem/actions/workflows/docs.yml/badge.svg">
        </a>

        <br clear="all" />

        <a href="https://github.com/unifyai/memory/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/memory/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/builder/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/builder/actions/workflows/nightly-tests.yml/badge.svg">
        </a>
        <a href="https://github.com/unifyai/models/actions?query=workflow%3Anightly-tests">
            <img class="dark-light" width="15%" style="float: left; margin: 0% 5%;" src="https://github.com/unifyai/models/actions/workflows/nightly-tests.yml/badge.svg">
        </a>

        <br clear="all" />

    </div>
    <br clear="all" />


Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
