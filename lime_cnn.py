from ultralytics import YOLO
from datasets import load_dataset
from PIL import Image
import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import quickshift
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import os

# Load the YOLO model
model = YOLO("models/yolov8SC.pt")  # Replace with your custom model path

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

# Ensure output directory exists
output_dir = "lime_explanations"
os.makedirs(output_dir, exist_ok=True)

# Define function to preprocess image for LIME
def preprocess_image(image):
    image = image.resize((640, 640))  # Resize to YOLO input size
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    return image

# Define function to get YOLO predictions for LIME
def predict_fn(images):
    images_tensor = [torch.tensor(img).permute(2, 0, 1).unsqueeze(0) for img in images]
    results = [model(img) for img in images_tensor]

    probs = []
    for res in results:
        if res[0].probs is not None:
            probs.append(res[0].probs.cpu().detach().numpy())  # Extract probabilities
        else:
            # If YOLO does not return probabilities, create fake ones
            fake_probs = np.array([0.5, 0.5])  # Assume equal chance for both classes
            probs.append(fake_probs)

    return np.array(probs)

# Process a subset of test images for LIME explanations
num_explanations = 5  # Change this to run on more images

for i, example in enumerate(test_split):
    image = example["image"]  # PIL Image
    ground_truth = example["malignancy"]  # True label

    # Preprocess image for LIME
    image_np = preprocess_image(image)

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=500,  # Speed optimization
        segmentation_fn=lambda img: quickshift(img, kernel_size=4, max_dist=200)
    )

    # Ensure a mask is generated
    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        top_label,
        positive_only=True,
        num_features=10,  # Force at least 10 features
        hide_rest=False
    )

    # Save and display the explanation
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(temp, mask))
    ax.set_title(f"LIME Explanation: Image {i+1} ({ground_truth})")
    ax.axis("off")

    # Save the explanation image
    lime_filename = os.path.join(output_dir, f"lime_image_{i+1}.png")
    plt.savefig(lime_filename, bbox_inches="tight")
    plt.show()

    print(f"LIME generated for image {i+1} (Ground Truth: {ground_truth}), saved at {lime_filename}")
    break

print("LIME explanations completed.")
