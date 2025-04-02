import torch
import numpy as np
import json
from lime import lime_image
from numba.cpython.slicing import slice_constructor_impl
from skimage.segmentation import quickshift, mark_boundaries, slic
from ultralytics import YOLO
from datasets import load_dataset
from PIL import Image
import matplotlib.pyplot as plt


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
    results = model(image_pil)

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



label_mapping = {
    "melanoma": "Malignant",
    "melanocytic_Nevi": "Malignant",
    "dermatofibroma": "Benign",
    "basal_cell_carcinoma": "Malignant",
    "vascular_lesions": "Benign",
    "actinic_keratoses": "Malignant",
    "benign_keratosis-like_lesions": "Benign"
}
# Load the dataset
dataset = load_dataset("marmal88/skin_cancer")
test_split = dataset["test"]

# Load the YOLO model
model = YOLO("yolo_weights/yolov8SC.pt")  # Replace with your model path

true_positives = [104, 53,208, 210,117,87,29]
# Process and explain 10 images
for index in true_positives:
    print(f"Processing image {index + 1}")
    image = test_split[index]["image"]
    ground_truth = test_split[index]["dx"]
    truth = label_mapping[ground_truth]

    # Transform image for SODEx
    image_np = transform_image(image)

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        lambda x: surrogate_classifier(x, model),
        top_labels=2,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=200)
        # segmentation_fn=lambda x: slic(x, n_segments=50, compactness=10)
    )


    # Get explanation mask
    image_explained, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False
    )
    image_explained2, mask2 = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False
    )

    # Resize the image and mask to 244x244
    image_explained2 = Image.fromarray(image_explained2).resize((224, 224))
    mask2 = Image.fromarray(mask2.astype(np.uint8)).resize((224, 224), Image.NEAREST)
    mask2 = np.array(mask2).astype(bool)

    # Overlay the explanation mask onto the original image
    overlay = mark_boundaries(np.array(image_explained2), mask2)

    # Plot only the overlay explanation
    plt.figure(figsize=(5, 5))
    plt.imshow(overlay)
    plt.title(f"Overlay Explanation - Image {index + 1}, label: {truth}")
    plt.axis("off")
    plt.savefig(f"results_lime/overlay_{index}.png")
    plt.show()

    print("Size of image: ", overlay.shape)
print("Done")