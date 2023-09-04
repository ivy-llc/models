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

VGG
===========

`VGG <https://arxiv.org/abs/1409.1556>`_ stands for Visual Geometry Group. It is a standard deep Convolutional Neural Network (CNN) architecture with multiple layers. 
The “deep” refers to the number of layers with VGG-16 or VGG-19 consisting of 16 and 19 convolutional layers

The VGG architecture is the basis of ground-breaking object recognition models. Developed as a deep neural network, 
the VGGNet also surpasses baselines on many tasks and datasets beyond ImageNet. Moreover, it is now still one of the most popular image recognition 
architectures.

Getting started
-----------------

.. code-block:: python

    import ivy
    from ivy_models.vgg import vgg16
    ivy.set_backend("torch")

    # Instantiate vgg16 model
    ivy_vgg16 = vgg16(pretrained=True)

The pretrained vgg16 model is now ready to be used, and is compatible with any other PyTorch code

Citation
--------

::

    @article{
      title={Very Deep Convolutional Networks for Large-Scale Image Recognition},
      author={Karen Simonyan and Andrew Zisserman},
      journal={arXiv preprint arXiv:1409.1556},
      year={2015}
    }


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
