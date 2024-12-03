import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

def inference(model, image_path, device, threshold=0.4, output_path="masked_output.jpg"):
    model.eval()

    # Load and preprocess the image
    image_cv = cv2.imread(image_path)
    image_tensor = torch.tensor(image_cv).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    # Filter boxes with high confidence
    boxes = boxes[scores >= threshold]

    # Mask the outside regions
    mask = np.zeros_like(image_cv, dtype=np.uint8)
    for box in boxes:
        x_min, y_min, x_max, y_max = box.astype(int)
        mask[y_min:y_max, x_min:x_max] = 255

    masked_image = cv2.bitwise_and(image_cv, mask)

    # Save the masked image
    cv2.imwrite(output_path, masked_image)
    print(f"Masked image saved as {output_path}")

    # Display the result
    plt.imshow(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
