from jedi.api import file_name
from ultralytics import YOLO
from datasets import load_dataset
from PIL import Image
import json
import numpy as np

# Load the YOLO model
model = YOLO("yolo_weights/yolov8SC.pt")  # Replace with your custom model path
file_name = "results/out_yolov8.json"
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

# Add this new label to the dataset (if needed)
test_split = test_split.add_column("malignancy", mapped_labels)

# Check unique values to confirm mapping
unique_mapped_labels = set(mapped_labels)
print(unique_mapped_labels)  # Should output {'malignant', 'benign'}

# Ground truth labels (already Malignant/Benign)
ground_truth_labels = test_split["malignancy"]

# Initialize accuracy tracking
correct_predictions = 0
total_predictions = 0

# Perform inference on the test set
for i, example in enumerate(test_split):
    image = example["image"]  # PIL Image
    ground_truth = ground_truth_labels[i]  # True label (Malignant/Benign)

    # Perform inference
    results = model(image)

    # Extract model's predicted Malignant/Benign label
    if len(results[0].boxes) > 0:
        predicted_label = results[0].names[int(results[0].boxes.cls[0].item())]  # Get predicted class
    else:
        predicted_label = "Unknown"  # If no detection

    # Update accuracy count
    if predicted_label == ground_truth:
        correct_predictions += 1
        print("TRUE")
    total_predictions += 1

    # Print progress
    print(f"Processed {i+1}/{len(test_split)}: Predicted = {predicted_label}, Actual = {ground_truth}")

# Compute final accuracy
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

# Save metrics
metrics = {
    "total_images": total_predictions,
    "correct_predictions": correct_predictions,
    "accuracy": accuracy
}

with open(file_name, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Malign/Benign Accuracy: {accuracy:.4f}")
print("Metrics saved to:", file_name)


