import cv2
import json
import os

# Đường dẫn
IMAGE_DIR = 'D:/segmentation/Segmentation/images/valid'          # Thư mục chứa ảnh gốc
OUTPUT_JSON = 'bounding_boxes.json'   # File JSON chứa bounding boxes
OUTPUT_DIR = 'output/visualized'      # Thư mục lưu ảnh có bounding boxes (nếu cần)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def visualize_bboxes(image_dir, bbox_json, output_dir):
    """
    Vẽ bounding boxes trên ảnh và hiển thị.
    Args:
        image_dir (str): Thư mục chứa ảnh gốc.
        bbox_json (str): File JSON chứa bounding boxes.
        output_dir (str): Thư mục lưu ảnh có bounding boxes (tùy chọn).
    """
    # Load bounding boxes
    with open(bbox_json, 'r') as f:
        annotations = json.load(f)

    for image_id, bboxes in annotations.items():
        # Load image
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        if not os.path.exists(image_path):
            print(f"Image {image_id}.jpg not found in {image_dir}. Skipping.")
            continue
        
        image = cv2.imread(image_path)
        
        # Draw bounding boxes
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        # Display image
        cv2.imshow('Bounding Boxes', image)
        cv2.waitKey(0)  # Press any key to proceed to the next image

        # Save the visualized image (optional)
        output_path = os.path.join(output_dir, f"{image_id}_bbox.jpg")
        cv2.imwrite(output_path, image)
        print(f"Saved {output_path}")

    cv2.destroyAllWindows()

# Thực hiện
visualize_bboxes(IMAGE_DIR, OUTPUT_JSON, OUTPUT_DIR)
