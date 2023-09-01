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

ConvNeXt
===========

ConvNeXt is a family of convolutional neural networks (CNNs) that were proposed in the paper `A ConvNet for the 2020s <https://arxiv.org/abs/2201.03545>`_. 
ConvNeXts are designed to be efficient and scalable, while still achieving high accuracy on image classification tasks.

ConvNeXts are based on a modular design, which allows them to be easily scaled to different sizes and configurations. 
The basic building block of a ConvNeXt is the "ConvNeXt block", which consists of a stack of convolutional layers with different dilation rates. 
The dilation rates are used to control the receptive field of the convolutional layers, 
which allows ConvNeXts to learn long-range dependencies in the input images.

Getting started
-----------------

.. code-block:: python

    import ivy
    from ivy_models.convnext import convnext_large
    ivy.set_backend("torch")

    # Instantiate convnext_large model
    ivy_convnext_large = convnext_large(pretrained=True)

    # Convert the Torch image tensor to an Ivy tensor and adjust dimensions
    img = ivy.asarray(ivy_convnext_large.permute((0, 2, 3, 1)), dtype="float32", device="gpu:0")

    # Compile the Ivy convnext_large model with the Ivy image tensor
    ivy_convnext_large.compile(args=(img,))

    # Pass the Ivy image tensor through the Ivy convnext_large model and apply softmax
    output = ivy.softmax(ivy_convnext_large(img))

    # Get the indices of the top 3 classes from the output probabilities
    classes = ivy.argsort(output[0], descending=True)[:3] 

    # Retrieve the logits corresponding to the top 3 classes
    logits = ivy.gather(output[0], classes) 


Citation
--------

::

    @article{
      title={A ConvNet for the 2020s},
      author={Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell and Saining Xie},
      journal={arXiv preprint arXiv:2201.03545},
      year={2022}
    }


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
