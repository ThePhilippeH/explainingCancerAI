import torch
import numpy as np
import cv2
from ultralytics import YOLO
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt

from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image, scale_cam_image

def preprocess_image(image):
    """Converts PIL Image to tensor and normalizes it."""
    image = image.convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (640, 640))  # Resize to YOLOv8 input size
    image = image / 255.0  # Normalize5
    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return image

# Load the YOLO model
model = YOLO("yolo_weights/yolov8SC.pt")  # Replace with your model path
model.eval()
print(model.model)

# Load dataset
dataset = load_dataset("marmal88/skin_cancer")
test_split = dataset["test"]

# Select a sample image from dataset
example = test_split[0]
image = example["image"]
rgb_img = np.array(image) / 255.0  # Normalize image for visualization
input_tensor = preprocess_image(image)

# Define target layers
target_layers = [model.model.model[10]]  # Adjust according to YOLOv8 model structure

# Apply EigenCAM
cam = EigenCAM(model, target_layers, task='od')
grayscale_cam = cam(rgb_img)[0, :, :]
cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# Display the result
plt.imshow(cam_image)
plt.title("Grad-CAM using EigenCAM")
plt.axis("off")
plt.show()