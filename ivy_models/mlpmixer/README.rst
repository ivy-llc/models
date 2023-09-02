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

MLP-Mixer
===========

`MLP-Mixer <https://arxiv.org/abs/2105.01601>`_ is based entirely on multi-layer perceptrons (MLPs), which are a type of neural network that consists of a stack of linear layers and 
non-linear activation functions.

The main idea behind MLP-Mixer is that MLPs can be used to learn spatial and channel mixing functions that can be used to extract features from images. 
MLP-Mixer achieves this by stacking two types of layers. These are the patch mixing layers and the channel mixing layers.
The patch mixing layers apply MLPs to each patch of the image, independently of the other patches. This allows MLP-Mixer to learn spatial mixing functions that can 
capture the relationships between different patches in the image.
The channel mixing layers on the otherhand apply MLPs to the entire image, across all channels. This allows MLP-Mixer to learn channel mixing functions that can 
capture the relationships between different channels in the image.


Getting started
-----------------

.. code-block:: python

    ! pip install huggingface_hub
    
    import ivy
    from ivy_models.mlpmixer import mlpmixer
    ivy.set_backend("torch")

    # Instantiate mlpmixer model
    ivy_mlpmixer = mlpmixer(pretrained=True)

The pretrained mlpmixer model is now ready to be used, and is compatible with any other PyTorch code

Citation
--------

::

    @article{
      title={MLP-Mixer: An all-MLP Architecture for Vision},
      author={
        Ilya Tolstikhin, Neil Houlsby, Alexander Kolesnikov, Lucas Beyer, Xiaohua Zhai, Thomas Unterthiner, 
        Jessica Yung, Andreas Steiner, Daniel Keysers, Jakob Uszkoreit, Mario Lucic and Alexey Dosovitskiy
      },
      journal={arXiv preprint arXiv:2105.01601},
      year={2021}
    }


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
