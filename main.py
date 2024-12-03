import torch
from dataset import FishDataset, get_transforms
from model import get_model
from train import train_model
from inference import inference
from utils import get_device

if __name__ == "__main__":
    # Paths
    image_dir = "D:/segmentation/Segmentation/images/valid"
    bbox_json = "bounding_boxes.json"
    image_path = "D:/segmentation/Segmentation/images/valid/9892_acanthopagrus_palmaris_f000060.jpg"

    # Initialize components
    device = get_device()
    print(f"Using device: {device}")

    dataset = FishDataset(image_dir, bbox_json, transforms=get_transforms())
    model = get_model(num_classes=2)

    # Train the model
    train_model(model, dataset, device)

    # Perform inference
    inference(model, image_path, device)
