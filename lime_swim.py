import os

from skimage.segmentation import quickshift, mark_boundaries
from transformers import pipeline
import kagglehub
import torch
import matplotlib.pyplot as plt
from lime import lime_image
import time
from PIL import Image
import numpy as np
from datasets import load_dataset

def transform_image(image):
    """Ensure image is in RGB format."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)
# Download latest version
# path = kagglehub.dataset_download("kmader/skin-cancer-mnist-ham10000")
# dataset = load_dataset(path)
dataset = load_dataset("marmal88/skin_cancer")
test_split = dataset["test"]
pipe = pipeline("image-classification", model="gianlab/swin-tiny-patch4-window7-224-finetuned-skin-cancer")
label_mapping = {
    "melanoma": "Malignant",
    "melanocytic_Nevi": "Malignant",
    "dermatofibroma": "Benign",
    "basal_cell_carcinoma": "Malignant",
    "vascular_lesions": "Benign",
    "actinic_keratoses": "Malignant",
    "benign_keratosis-like_lesions": "Benign"
}
label_list = ["Melanoma",
              "Melanocytic-nevi",
    "Dermatofibroma",
    "Basal-cell-carcinoma",
    "Vascular-lesions",
    "Actinic-keratoses",
    "Benign-keratosis-like-lesions"]
def surrogate_classifier(images_np, pipeline):
    """Generate a surrogate classifier's output based on detected objects."""
    outputs = []
    for image_np in images_np:
        try:
            image_input = Image.fromarray(image_np)
            scores = pipeline.predict(image_input)
            output = np.zeros(len(label_list))  # Initialize output for all classes
            for score in scores:
                class_id = label_list.index(score["label"])
                output[class_id] = score["score"] # Assign confidence scores to detected classes
            outputs.append(output)
        except Exception as e:
            print(f"Error processing image: {e}")
            outputs.append(np.zeros(len(label_list)))  # Return zero scores if error
    return np.array(outputs)
test = test_split[0]["image"]
result = pipe.predict(test)
print(result)
# Ensure output directory exists
output_dir = "lime_explanations"
os.makedirs(output_dir, exist_ok=True)
for index in range(10):
    print(f"Processing image {index + 1}/10")
    image = test_split[index]["image"]
    ground_truth = test_split[index]["dx"]
    truth = label_mapping[ground_truth]

    # Transform image for SODEx
    image_np = transform_image(image)

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_np,
        lambda x: surrogate_classifier(x, pipe),
        top_labels=5,
        hide_color=0,
        num_samples=1000,
        segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=200)
    )
    image_explained, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False)
    overlay = mark_boundaries(image_np, mask)

    # Plot only the overlay explanation
    # Save and display the explanation
    fig, ax = plt.subplots()
    ax.imshow(mark_boundaries(image_explained, mask))
    ax.set_title(f"LIME Explanation: Image {index + 1} ({ground_truth})")
    ax.axis("off")

    # Save the explanation image
    lime_filename = os.path.join(output_dir, f"lime_image_{index + 1}.png")
    plt.savefig(lime_filename, bbox_inches="tight")
    plt.show()

print("Processing complete.")
#
