import torch
from torchvision.models import googlenet, get_model_weights
import torchvision.transforms as transforms
from PIL import Image

def generate_gt_inference_for(test_image_path=None):
    # Initialize model with the GoogLeNet_Weights.IMAGENET1K_V1 weight
    weights = get_model_weights("GoogLeNet")
    model = googlenet(weights=weights)
    model.eval()

    # Load and preprocess the cat image
    # image_path = test_image_path
    image_path = 'F:\Github_dsc\models_Ivy_Sark42\images\cat.jpg'
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

    # return predicted_class


generate_gt_inference_for()
