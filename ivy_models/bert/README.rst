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

Bert
===========

`BERT <https://arxiv.org/abs/1810.04805>`_ short for Bidirectional Encoder Representations from Transformers, differentiates itself from 
recent language representation models by its focus on pretraining deep bidirectional representations from unannotated text. 
This approach involves considering both left and right context in all layers.
Consequently, the pretrained BERT model can be enhanced with just a single additional output layer to excel in various tasks, 
such as question answering and language inference. This achievement is possible without extensive modifications to task-specific architecture.

Getting started
-----------------

.. code-block:: python

    import ivy
    ivy.set_backend("torch")

    # Instantiate Bert
    ivy_bert = ivy_models.bert_base_uncased(pretrained=True)

    # Convert the input data to Ivy tensors
    ivy_inputs = {k: ivy.asarray(v.numpy()) for k, v in inputs.items()}

    # Compile the Ivy BERT model with the Ivy input tensors
    ivy_bert.compile(kwargs=ivy_inputs)

    # Pass the Ivy input tensors through the Ivy BERT model and obtain the pooler output
    ivy_output = ivy_bert(**ivy_inputs)['pooler_output']


See `this demo <https://github.com/unifyai/demos/blob/main/examples_and_demos/bert_demo.ipynb>`_ for more usage example.

Citation
--------

::

    @article{
      title={BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding},
      author={Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova},
      journal={arXiv preprint arXiv:1810.04805},
      year={2019}
    }


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
