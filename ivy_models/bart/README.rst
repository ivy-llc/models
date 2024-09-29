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

BART
===========

`BART <https://arxiv.org/abs/1910.13461>`_  also known as Bidirectional Autoencoder Representations from Transformers is a denoising autoencoder for pretraining 
sequence-to-sequence models. It is trained by corrupting text with an arbitrary noising function, and learning a model to reconstruct the original text. 
BART uses a standard Transformer-based neural machine translation architecture, which consists of a bidirectional encoder and a left-to-right decoder.

The encoder takes the corrupted text as input and produces a sequence of hidden states. The decoder then takes these hidden states as input and predicts the original text, 
one token at a time. The model is trained to minimize the negative log likelihood of the original text.
BART can be used for a variety of natural language processing tasks, including text generation, translation, and comprehension. 
It has been shown to achieve state-of-the-art results on a number of these tasks


Getting started
-----------------

.. code-block:: python

    import ivy
    ivy.set_backend("torch")
    from ivy_models.bart import BartModel
    from ivy_models.bart.config_bart import BartConfig

    # Instantiate bart model
    ivy_bart = BartModel(BartConfig)

The pretrained bart model is now ready to be used, and is compatible with any other PyTorch code

Citation
--------

::

    @article{
      title={BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension},
      author={Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer},
      journal={arXiv preprint arXiv:1910.13461},
      year={2019}
    }


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
