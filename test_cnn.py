import torch
from PIL import Image
import cv2

# Load YOLOv5 model from torch hub
model = torch.hub.load("ultralytics/yolov5", "custom", path="weights.pt")  # Load your custom weights

# Perform inference on an image
img = Image.open("path_to_your_image.jpg")  # Replace with your image path

# Run inference
results = model(img)

# Display results
results.show()  # Opens the image with bounding boxes
results.print()  # Prints detection results to the console

# Alternatively, save the results
results.save("output.jpg")  # Saves the image with bounding boxes

# Access detection details
detections = results.xyxy[0].numpy()  # Get detections as a numpy array
print(detections)  # Each row contains [xmin, ymin, xmax, ymax, confidence, class]