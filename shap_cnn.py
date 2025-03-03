import shap
import torch
from PIL import Image
import numpy as np
import os
from ultralytics import YOLO
from datasets import load_dataset
import matplotlib.pyplot as plt
import tensorflow

# Load YOLO model
model = YOLO("yolo_weights/yolov8SC.pt")  # Replace with your custom model path

# Load the dataset
dataset = load_dataset("marmal88/skin_cancer")
label_mapping = {
    "melanoma": "Malignant",
    "melanocytic_Nevi": "Malignant",
    "dermatofibroma": "Benign",
    "basal_cell_carcinoma": "Malignant",
    "vascular_lesions": "Benign",
    "actinic_keratoses": "Malignant",
    "benign_keratosis-like_lesions": "Benign"
}

# Apply the mapping to the 'dx' column
test_split = dataset["test"]
mapped_labels = [label_mapping[label] for label in test_split["dx"]]
test_split = test_split.add_column("malignancy", mapped_labels)

# Define a function to preprocess images for SHAP (consistent with YOLO input)
def preprocess_image(image):
    # Resize image to 640x640 as expected by YOLO
    image = image.resize((640, 640))
    image = np.array(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    return image

# Define a function to get YOLO predictions for SHAP
def predict_fn(images):
    images_tensor = [torch.tensor(img).permute(2, 0, 1).unsqueeze(0) for img in images]
    results = [model(img) for img in images_tensor]

    probs = []
    for res in results:
        if len(res[0].boxes) > 0:  # If there are detections
            # Extract class probabilities (for each object detected)
            # Assuming we are interested in the first detected object (can be adjusted for multiple detections)
            probs.append(res[0].probs[0].cpu().detach().numpy())  # Extract the first detection's class probabilities
        else:
            # If no detections, return a neutral prediction (50-50 probability for both classes)
            fake_probs = np.array([0.5, 0.5])  # No detection, assume equal probability
            probs.append(fake_probs)

    return np.array(probs)

# Ensure output directory exists
output_dir = "shap_explanations"
os.makedirs(output_dir, exist_ok=True)

# SHAP Explainer
# We will use a dummy tensor for the explainer initialization since we only need it to define the input shape
dummy_input = torch.ones(1, 3, 640, 640)  # Dummy input for model (640x640 image, 3 channels)

# Wrap the model in a function that returns the class probabilities
def model_wrapper(input_tensor):
    with torch.no_grad():
        results = model(input_tensor)
        if len(results[0].boxes) > 0:
            return results[0].probs[0].unsqueeze(0)  # Return the class probabilities for the first detection
        else:
            return torch.tensor([[0.5, 0.5]])  # Return neutral probabilities if no detection

explainer = shap.GradientExplainer(model_wrapper, dummy_input)  # Initialize explainer

# Process a subset of test images for SHAP explanations
num_explanations = 5  # Change this to run on more images
for i, example in enumerate(test_split):
    image = example["image"]  # PIL Image
    ground_truth = example["malignancy"]  # True label

    # Preprocess image for SHAP
    image_np = preprocess_image(image)

    # Generate SHAP explanation for the image
    shap_values = explainer.shap_values(torch.tensor(image_np).unsqueeze(0).permute(0, 3, 1, 2))

    # Extract SHAP values for the first class
    shap_map = shap_values[0].cpu().detach().numpy()[0]  # Get SHAP values for the first class

    # Generate a heatmap to visualize SHAP values
    shap_image = np.abs(shap_map).sum(axis=0)  # Summing the absolute values across channels
    shap_image = np.clip(shap_image, 0, 1)  # Ensure values are between 0 and 1

    # Plot the SHAP values
    plt.imshow(shap_image, cmap='hot')
    plt.colorbar()
    plt.title(f"SHAP Explanation for Image {i + 1} (Ground Truth: {ground_truth})")
    plt.axis("off")

    # Save SHAP explanation
    shap_filename = os.path.join(output_dir, f"shap_image_{i + 1}.png")
    plt.savefig(shap_filename, bbox_inches="tight")
    plt.show()

    print(f"SHAP explanation generated for image {i + 1} (Ground Truth: {ground_truth}), saved at {shap_filename}")

    # Uncomment the break if you want to process only one image for testing
    break

print("SHAP explanations completed.")