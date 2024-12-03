import os
import torch
from PIL import Image
import json

class FishDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, bbox_json, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        with open(bbox_json, 'r') as f:
            self.annotations = json.load(f)
        self.image_ids = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        img = Image.open(img_path).convert("RGB")

        # Get bounding boxes
        bboxes = torch.tensor(self.annotations[image_id], dtype=torch.float32)

        # Define target
        target = {
            "boxes": bboxes,  # Bounding boxes
            "labels": torch.ones((len(bboxes),), dtype=torch.int64),  # Class '1' for all (fish)
            "image_id": torch.tensor([idx])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target

# Example usage
image_dir = "Segmentation/images/valid"
bbox_json = "bounding_boxes.json"
dataset = FishDataset(image_dir, bbox_json)
print(f"Dataset size: {len(dataset)}")
