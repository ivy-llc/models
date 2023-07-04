import os
import ivy
import pytest
import numpy as np
from ivy_models.googlenet import inceptionNet_v1
from ivy_models_tests import helpers

import torch
from torchvision.models import googlenet, get_model_weights
import torchvision.transforms as transforms
from PIL import Image

def generate_gt_inference_for(test_image_path):
    # Initialize model with the GoogLeNet_Weights.IMAGENET1K_V1 weight
    weights = get_model_weights("GoogLeNet")
    model = googlenet(weights=weights)
    model.eval()

    # Load and preprocess the cat image
    image_path = test_image_path
    # image_path = 'F:\Github_dsc\models_Ivy_Sark42\images\cat.jpg'
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # 3-3 values are calculated mean and std on imagenet dataset for 3 channels each
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class labels and logits
    _, predicted_indices = torch.topk(output, k=3, dim=1)
    predicted_logits = output[0, predicted_indices[0]]

    # Convert tensors to Python lists
    predicted_classes = predicted_indices[0].tolist()
    predicted_logits = predicted_logits.tolist()

    # # Print the predicted class labels and logits
    # print("Top 3 predicted classes:")
    # for class_idx, logit in zip(predicted_classes, predicted_logits):
    #     print(f"Class: {class_idx}, Logit: {logit}")

    return predicted_classes, predicted_logits

@pytest.mark.parametrize("batch_shape", [[1]])
@pytest.mark.parametrize("load_weights", [False, True])
def test_inception_v1_img_classification(device, f, fw, batch_shape, load_weights):
    """Test Inception-V1 image classification."""
    num_classes = 1000
    this_dir = os.path.dirname(os.path.realpath(__file__))

    # Load image
    img = helpers.load_and_preprocess_img(
        os.path.join(this_dir, "..", "..", "images", "cat.jpg"), 256, 224
    )

    # Create model
    model = inceptionNet_v1(pretrained=load_weights)

    # Perform inference
    output = model(img)

    # Cardinality test
    # weights == specificity of the specific model
    # (becoz arch of the model decided the no of trainable parameters in that model
    # ie, weigths and biases.)

    # probabs of all 1000 classes for each image in the batch, ie, it's a 2d array
    assert output.shape == tuple([ivy.to_scalar(batch_shape), num_classes])

    # Value test
    if load_weights:
        predicted_classes, predicted_logits = generate_gt_inference_for(img)
        # 1) comparing the indices of 3 highest probabs having classes from gt and cal output arrays
        # pick probab values for 1000 classes for the first image only from the batch of size (batch_shape)
        output = output[0]
        # indicies for the 3 highest probab classes from unsorted array are [282, 285, 281] in gt_output
        true_indices = ivy.array(predicted_classes).sort()
        # indicies for the 3 highest probab classes from unsorted array
        calc_indices = ivy.argsort(output, descending=True)[:3].sort()
        assert np.array_equal(true_indices, calc_indices)

        # 2) calculating probab values for 3 classes from gt and calc output arrays
        true_logits = ivy.array(predicted_logits).sort()
        calc_logits = ivy.take_along_axis(output, calc_indices, 0)
        assert np.allclose(true_logits, calc_logits, rtol=0.5)
