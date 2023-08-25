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

AlexNet
===========

AlexNet competed in the ImageNet Large Scale Visual Recognition Challenge on September 30, 2012. 

The network achieved a top-5 error of 15.3%, more than 10.8 percentage points lower than that of the runner up. 
The original paper’s primary result was that the depth of the model was essential for its high performance, which was computationally expensive, 
but made feasible due to the utilization of graphics processing units (GPUs) during training.

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
    ivy.set_backend("torch")
    model = alexnet()

The pretrained AlexNet model is now ready to be used, and is compatible with any other PyTorch code.
See `this demo <https://unify.ai/demos/examples_and_demos/alexnet_demo.html>`_ for more usage example.


References
-----------------------------
1. `One weird trick for parallelizing convolutional neural networks <https://arxiv.org/abs/1404.5997>`





Citation
--------

::

    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
