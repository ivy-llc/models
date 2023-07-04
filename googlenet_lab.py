import torch
from torchvision.models import googlenet, get_model_weights
import torchvision.transforms as transforms
from PIL import Image

def generate_gt_inference_for(input_tensor):
    # Initialize model with the GoogLeNet_Weights.IMAGENET1K_V1 weight
    weights = get_model_weights("GoogLeNet")
    model = googlenet(weights=weights)
    model.eval()

    # Load and preprocess the cat image
    image_path = 'F:\Github_dsc\models_Ivy_Sark42\images\cat.jpg'
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)

    # Make the prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Get the predicted class label
    _, predicted_idx = torch.max(output, 1)
    predicted_class = predicted_idx.item()

    return predicted_class
    # Print the predicted class label
    # print(f"Predicted class for the cat image: {predicted_class}")
