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

RESNET
===========

`RESNET <https://arxiv.org/abs/1512.03385>`_ also known as Residual Network is a deep learning model utilized in computer vision tasks. 
It's a type of Convolutional Neural Network (CNN) architecture designed to accommodate a large number of convolutional layers, 
possibly ranging from hundreds to thousands. ResNet addresses the challenge of diminishing gradients, a problem encountered during training, 
by introducing a concept called "skip connections." These connections enable the network to skip over several initial layers, 
which consist of identity mappings (convolutional layers with no immediate impact). 
This results in faster initial training by condensing the network into fewer layers.

During subsequent retraining, the network expands to its full depth, and the sections of the network that weren't initially utilized, 
called "residual parts," are given the opportunity to explore the features present in the input image more comprehensively.

Getting started
-----------------

.. code-block:: python

    import ivy
    from ivy_models.resnet import resnet_34
    ivy.set_backend("torch")

    # Instantiate Resnet_34 model
    ivy_resnet_34 = resnet_34(pretrained=True)

    # Convert the Torch image tensor to an Ivy tensor and adjust dimensions
    img = ivy.asarray(torch_img.permute((0, 2, 3, 1)), dtype="float32", device="gpu:0")

    # Compile the Ivy Resnet_34 model with the Ivy image tensor
    ivy_resnet_34.compile(args=(img,))

    # Pass the Ivy image tensor through the Ivy Resnet_34 model and apply softmax
    output = ivy.softmax(ivy_resnet_34(img))

    # Get the indices of the top 3 classes from the output probabilities
    classes = ivy.argsort(output[0], descending=True)[:3] 

    # Retrieve the logits corresponding to the top 3 classes
    logits = ivy.gather(output[0], classes) 


    print("Indices of the top 3 classes are:", classes)
    print("Logits of the top 3 classes are:", logits)
    print("Categories of the top 3 classes are:", [categories[i] for i in classes.to_list()])

    `Indices of the top 3 classes are: ivy.array([282, 281, 285], dev=gpu:0)`
    `Logits of the top 3 classes are: ivy.array([0.85072654, 0.13506058, 0.00688287], dev=gpu:0)`
    `Categories of the top 3 classes are: ['tiger cat', 'tabby', 'Egyptian cat']`

See `this demo <https://unify.ai/demos/examples_and_demos/resnet_demo.html>`_ for more usage example.

Citation
--------

::

    @article{
      title={Deep Residual Learning for Image Recognition},
      author={Kaiming He, Xiangyu Zhang, Shaoqing Ren and Jian Sun},
      journal={arXiv preprint arXiv:1512.03385},
      year={2015}
    }


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
