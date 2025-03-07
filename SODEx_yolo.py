import torch
import numpy as np
import json
from lime import lime_image
from skimage.segmentation import quickshift
from ultralytics import YOLO
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries



def transform_image(image):
    """Ensure image is in RGB format."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def yolo_predict(image_np, model):
    """Performs YOLO inference and returns object detection results."""
    if image_np.shape[-1] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3)")

    image_pil = Image.fromarray(image_np.astype(np.uint8))  # Ensure proper dtype
    results = model(image_pil, conf = 0.05)

    if len(results[0].boxes) > 0:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        classes = results[0].boxes.cls.cpu().numpy()  # Class IDs
        scores = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        return boxes, classes, scores
    else:
        return [], [], []  # No detections


def surrogate_classifier(images_np, model):
    """Generate a surrogate classifier's output based on detected objects."""
    outputs = []
    for image_np in images_np:
        try:
            boxes, classes, scores = yolo_predict(image_np, model)
            output = np.zeros(len(model.names))  # Initialize output for all classes
            for class_id, score in zip(classes, scores):
                output[int(class_id)] = score  # Assign confidence scores to detected classes
            outputs.append(output)
        except Exception as e:
            print(f"Error processing image: {e}")
            outputs.append(np.zeros(len(model.names)))  # Return zero scores if error
    return np.array(outputs)


# Load the dataset
dataset = load_dataset("marmal88/skin_cancer")
test_split = dataset["test"]

# Load the YOLO model
model = YOLO("yolo_weights/yolov8SC.pt")  # Replace with your model path

# Select an image from the test set
index = 0  # Change index to analyze different images
image = test_split[index]["image"]
ground_truth = test_split[index]["dx"]

# Transform image for SODEx
image_np = transform_image(image)

# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image_np, lambda x: surrogate_classifier(x, model),
                                         top_labels=2, hide_color=0, num_samples=2000,
                                         segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=200))

# Get explanation mask
image_explained, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=10,
                                                       hide_rest=False)


# Overlay the explanation mask onto the original image
overlay = mark_boundaries(image_np, mask)

# Plot the original image and explanation overlay
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(image_np)
ax[0].set_title("Original Image")
ax[1].imshow(image_explained)
ax[1].set_title("SODEx Explanation")
ax[2].imshow(overlay)
ax[2].set_title("Overlay Explanation")
plt.show()
# Save explanation
# explanation_data = {
#     "image_index": index,
#     "ground_truth": ground_truth,
#     "top_label": explanation.top_labels[0]
# }

# with open("sodex_explanation.json", "w") as f:
#     json.dump(explanation_data, f, indent=4)
#
# print("SODEx explanation saved to sodex_explanation.json")
