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

U-Net
===========

The `U-Net <https://arxiv.org/abs/1505.04597>`_  architecture and training approach effectively leverage data augmentation to make the most of 
available annotated samples, even with limited data. The design features a contracting path for context capture and a symmetric expanding path 
for precise localization. Notably, this UNET network achieves superior performance with minimal images during end-to-end training. 
It surpasses previous methods, including a sliding-window convolutional network, in the ISBI challenge for segmenting neuronal structures in 
electron microscopic stacks. Additionally, the same UNET model excels in the ISBI cell tracking challenge for transmitted light microscopy images 
(phase contrast and DIC), showcasing its versatility. Furthermore, the network demonstrates remarkable speed, capable of segmenting a 512x512 image 
in under a second using a modern GPU.

Getting started
-----------------

.. code-block:: python

    import ivy
    import ivy_models
    ivy.set_backend("torch")

    # load the unet model from ivy_models
    ivy_unet = ivy_models.unet_carvana(n_channels=3, n_classes=2, pretrained=True)

    # Preprocess image with preprocess function
    from PIL import Image
    !wget https://raw.githubusercontent.com/unifyai/models/master/images/car.jpg
    filename = "car.jpg"
    full_img = Image.open(filename)
    torch_img = torch.from_numpy(preprocess(None, full_img, 0.5, False)).unsqueeze(0).to("cuda")

    # Convert to ivy
    ivy.set_backend("torch")
    img = ivy.asarray(torch_img.permute((0, 2, 3, 1)), dtype="float32", device="gpu:0")
    img_numpy = img.cpu().numpy()

    # Compile the forward pass
    ivy_unet.compile(args=(img,))

    # Generating the mask 
    output = ivy_unet(img)
    output = ivy.interpolate(output.permute((0, 3, 1, 2)), (full_img.size[1], full_img.size[0]), mode="bilinear")
    mask = output.argmax(axis=1)
    mask = ivy.squeeze(mask[0], axis=None).to_numpy()
    result = mask_to_image(mask, [0,1])


See `this demo <https://github.com/unifyai/demos/blob/main/examples_and_demos/image_segmentation_with_ivy_unet.ipynb>`_ for more usage example.

Citation
--------

::

    @article{
      title={U-Net: Convolutional Networks for Biomedical Image Segmentation},
      author={Olaf Ronneberger, Philipp Fischer and Thomas Brox},
      journal={arXiv preprint arXiv:1505.04597},
      year={2015}
    }


    @article{lenton2021ivy,
      title={Ivy: Templated deep learning for inter-framework portability},
      author={Lenton, Daniel and Pardo, Fabio and Falck, Fabian and James, Stephen and Clark, Ronald},
      journal={arXiv preprint arXiv:2102.02886},
      year={2021}
    }
