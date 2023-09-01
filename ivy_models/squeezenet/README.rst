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

SqueezeNet
===========

`SqueezeNet <https://arxiv.org/abs/1602.07360>`_ is a convolutional neural network (CNN) architecture that is designed to be efficient and lightweight, 
while still achieving high accuracy on image classification tasks. SqueezeNet achieves its efficiency by using a number of techniques, including:
fire modules which are the basic building blocks of SqueezeNet, depthwise separable convolutions which are used in the Fire modules, and 
global average pooling which is used instead of fully connected layers.

SqueezeNet has been shown to achieve state-of-the-art results on a variety of image classification tasks, including ImageNet, CIFAR-10, and CIFAR-100. 
It is also relatively efficient, making it suitable for deployment on mobile devices. Some of the key features of SqueezeNet includes:
efficiency due to it's significantly smaller number of parameters, lightweight due to it's relatively small model size of less than 0.5MB and accuracy

Getting started
-----------------

.. code-block:: python

    import ivy
    from ivy_models.squeezenet import squeezenet1_0
    ivy.set_backend("torch")

    # Instantiate squeezenet1_0 model
    ivy_squeezenet1_0 = squeezenet1_0(pretrained=True)

    # Convert the Torch image tensor to an Ivy tensor and adjust dimensions
    img = ivy.asarray(torch_img.permute((0, 2, 3, 1)), dtype="float32", device="gpu:0")

    # Compile the Ivy squeezenet1_0 model with the Ivy image tensor
    ivy_squeezenet1_0.compile(args=(img,))

    # Pass the Ivy image tensor through the Ivy squeezenet1_0 model and apply softmax
    output = ivy.softmax(ivy_squeezenet1_0(img))

    # Get the indices of the top 3 classes from the output probabilities
    classes = ivy.argsort(output[0], descending=True)[:3] 

    # Retrieve the logits corresponding to the top 3 classes
    logits = ivy.gather(output[0], classes) 

    print("Indices of the top 3 classes are:", classes)
    print("Logits of the top 3 classes are:", logits)
    print("Categories of the top 3 classes are:", [categories[i] for i in classes.to_list()])


Citation
--------

::

    @article{
      title={SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size},
      author={Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally and Kurt Keutzer},
      journal={arXiv preprint arXiv:1602.07360},
      year={2016}
    }


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
