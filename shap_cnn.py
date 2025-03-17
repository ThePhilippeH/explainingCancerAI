import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import shap
import matplotlib.pyplot as plt
from datasets import load_dataset

# Step 1: Casting the image to a PyTorch tensor
class Numpy2TorchCaster(nn.Module):
    def forward(self, x):
        return torch.tensor(x, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

# Step 2: Core model (e.g., YOLO or any object detection model)
class CoreModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

# Step 3: Non-Maximum Suppression (NMS) and score calculation
class OD2Score(nn.Module):
    def __init__(self, target_class, iou_threshold=0.5):
        super().__init__()
        self.target_class = target_class
        self.iou_threshold = iou_threshold

    def forward(self, detections):
        boxes, scores, classes = detections
        # Apply NMS and filter for the target class
        filtered_boxes, filtered_scores = self._apply_nms(boxes, scores, classes)
        return filtered_boxes, filtered_scores

    def _apply_nms(self, boxes, scores, classes):
        # Implement NMS logic here
        # For simplicity, we assume the model already applies NMS
        return boxes, scores
# Step 4: Superpixel segmentation
class SuperPixler(nn.Module):
    def __init__(self, grid_size=(8, 8)):
        super().__init__()
        self.grid_size = grid_size

    def forward(self, x, active_superpixels=None):
        if active_superpixels is None:
            # If no active_superpixels are provided, assume all are active
            active_superpixels = np.ones((self.grid_size[0] * self.grid_size[1],))
        # Replace inactive superpixels with the mean color
        output = self._apply_superpixels(x, active_superpixels)
        return output

    def _apply_superpixels(self, x, active_superpixels):
        # Reshape the image into superpixels
        h, w, c = x.shape
        grid_h, grid_w = self.grid_size
        superpixel_h, superpixel_w = h // grid_h, w // grid_w

        # Create a mask for active superpixels
        mask = active_superpixels.reshape(grid_h, grid_w, 1, 1, 1)
        mask = np.kron(mask, np.ones((superpixel_h, superpixel_w, 1)))  # Use 1 channel for mask
        mask = mask[:, :, :h, :w, :]  # Trim to match the image size
        mask = mask.reshape(h, w, 1)  # Reshape to (h, w, 1) for broadcasting

        # Replace inactive superpixels with the mean color
        mean_color = np.mean(x, axis=(0, 1), keepdims=True)
        output = x * mask + mean_color * (1 - mask)
        return output



# Step 5: Combine all layers into a single model
class SuperPixelModel(nn.Module):
    def __init__(self, model, target_class, grid_size=(8, 8)):
        super().__init__()
        self.super_pixler = SuperPixler(grid_size=grid_size)
        self.numpy2torch = Numpy2TorchCaster()
        self.core_model = CoreModel(model)
        self.od2score = OD2Score(target_class=target_class)
        self.grid_size = grid_size

    def forward(self, active_superpixels, image=None):
        if image is None:
            raise ValueError("Image must be provided for superpixel segmentation.")
        # Apply superpixel segmentation
        x = self.super_pixler(image, active_superpixels)
        # Convert to PyTorch tensor
        x = self.numpy2torch(x)
        # Apply the core model
        detections = self.core_model(x)
        # Apply NMS and calculate the score
        boxes, scores = self.od2score(detections)
        return scores



# Load the YOLO model (or any object detection model)
from ultralytics import YOLO
model = YOLO("yolo_weights/yolov8SC.pt")  # Replace with your custom model path

# Define the target class (e.g., "person")
target_class = "Malignant"

# Step 6: Create the superpixel model
super_pixel_model = SuperPixelModel(model, target_class, grid_size=(8, 8))

# Step 7: Prepare the input image
def preprocess_image(image_path):
    image = image_path.convert("RGB")
    image = image.resize((640, 640))  # Resize to the input size expected by YOLO
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image

# Load an example image
# Load an example image
dataset = load_dataset("marmal88/skin_cancer")
testset = dataset["test"]
image_test = testset[10]["image"]
image = preprocess_image(image_test)


# Step 8: Define background and input superpixels
background_super_pixel = np.zeros((1, 8 * 8))  # Reshape to 2D array with 1 row
image_super_pixel = np.ones((1, 8 * 8))  # Reshape to 2D array with 1 row

# Step 9: Initialize SHAP KernelExplainer
explainer = shap.KernelExplainer(
    lambda active_superpixels: super_pixel_model(active_superpixels, image=image),
    background_super_pixel
)

# Step 10: Compute SHAP values
shap_values = explainer.shap_values(image_super_pixel, nsamples=3000)

# Step 11: Visualize SHAP values
shap.image_plot(shap_values, -image.numpy(), show=False)

# Save the SHAP plot
plt.savefig("shap_explanation.png")
plt.show()

print("SHAP explanation saved to shap_explanation.png")

