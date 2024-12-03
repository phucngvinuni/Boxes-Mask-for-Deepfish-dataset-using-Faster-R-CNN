import torch
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import cv2
import os
import numpy as np
from PIL import Image
import torchvision
# Function to load the model
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# Function to mask the background
def mask_background(image_path, model, device, threshold=0.4):
    # Load the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)
    
    # Run inference
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)

    # Extract bounding boxes and scores
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    # Filter boxes by confidence threshold
    boxes = boxes[scores >= threshold]

    # Load the image in OpenCV format
    image_cv = cv2.imread(image_path)
    mask = np.zeros_like(image_cv, dtype=np.uint8)

    # Create a mask for detected boxes
    for box in boxes:
        x_min, y_min, x_max, y_max = box.astype(int)
        mask[y_min:y_max, x_min:x_max] = 255  # White inside the bounding boxes

    # Apply the mask
    masked_image = cv2.bitwise_and(image_cv, mask)
    return masked_image

# Main function
def main():
    # Path to your saved model weights
    model_weights = "D:/segmentation/model_epoch_3.pth"
    
    # Initialize model
    num_classes = 2  # Background + Fish
    model = get_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the saved weights
    model.load_state_dict(torch.load(model_weights))
    print("Model loaded successfully!")

    # Directory containing images
    input_dir = "D:/segmentation/deepjscc"
    output_dir = "D:/segmentation/Masked_Imagesdeep"
    os.makedirs(output_dir, exist_ok=True)

    # Process all images
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name)

        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Generate black background image
            masked_image = mask_background(image_path, model, device)
            
            # Save the output
            cv2.imwrite(output_path, masked_image)
            print(f"Processed and saved: {output_path}")

if __name__ == "__main__":
    main()
