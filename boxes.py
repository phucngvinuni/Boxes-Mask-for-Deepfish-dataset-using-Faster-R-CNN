import cv2
import os
import json
import numpy as np
from tqdm import tqdm

# Đường dẫn dữ liệu
MASK_DIR = 'D:/segmentation/Segmentation/masks/valid'  # Thư mục chứa segmentation masks
OUTPUT_JSON = 'bounding_boxes.json'      # File lưu bounding boxes

def mask_to_bboxes(mask):
    """
    Chuyển từ segmentation mask sang bounding boxes.
    Args:
        mask (numpy array): Mask nhị phân (0, 255).
    Returns:
        list: Danh sách các bounding boxes [x_min, y_min, x_max, y_max].
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bboxes.append([x, y, x + w, y + h])
    return bboxes

def process_masks(mask_dir, output_json):
    """
    Duyệt qua tất cả các mask, chuyển sang bounding boxes và lưu kết quả.
    Args:
        mask_dir (str): Thư mục chứa segmentation masks.
        output_json (str): Đường dẫn file lưu bounding boxes.
    """
    annotations = {}
    for mask_file in tqdm(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Chuyển mask sang bounding boxes
        bboxes = mask_to_bboxes(mask)

        # Lưu bounding boxes vào dictionary
        image_id = os.path.splitext(mask_file)[0]  # ID ảnh
        annotations[image_id] = bboxes

    # Lưu kết quả vào file JSON
    with open(output_json, 'w') as f:
        json.dump(annotations, f, indent=4)

# Thực hiện
process_masks(MASK_DIR, OUTPUT_JSON)
print(f"Bounding boxes saved to {OUTPUT_JSON}")
