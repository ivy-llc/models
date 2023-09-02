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

ViT
===========

Vision Transformer `(ViT) <https://arxiv.org/abs/2010.11929>`_ is a neural network architecture for image classification that is based on the Transformer architecture, 
which was originally developed for natural language processing tasks. However, 
ViT replaces the convolution layers in a convolutional neural network (CNN) with self-attention layers.

The main idea behind ViT is that an image can be represented as a sequence of image patches, and that these patches can be processed by a Transformer 
in the same way that words are processed by a Transformer in a natural language processing task. 
To do this, ViT first divides the image into a grid of image patches. Each patch is then flattened into a vector, 
and these vectors are then stacked together to form a sequence. This sequence is then passed to a Transformer, 
which learns to attend to different patches in the image in order to classify the image.


Getting started
-----------------

.. code-block:: python

    import ivy
    from ivy_models.vit import vit_h_14
    ivy.set_backend("torch")

    # Instantiate vit_h_14 model
    ivy_vit_h_14 = vit_h_14(pretrained=True)

The pretrained vit_h_14 model is now ready to be used, and is compatible with any other PyTorch code

Citation
--------

::

    @article{
      title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
      author={
        Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, 
        Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit and Neil Houlsby
      },
      journal={arXiv preprint arXiv:2010.11929},
      year={2021}
    }


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
