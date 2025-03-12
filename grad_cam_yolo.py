from YOLOv8_Explainer import yolov8_heatmap, display_images
import matplotlib.pyplot as plt
from datasets import load_dataset
# Initialize the YOLOv8 Explainer with Grad-CAM
model = yolov8_heatmap(
    weight="yolo_weights/yolov8SC.pt",  # Path to your YOLOv8 model weights
    conf_threshold=0.4,  # Confidence threshold for detections
    method="GradCAM",  # Use Grad-CAM instead of EigenCAM
    layer=[10, 12, 14, 16, 18, -3],  # Target layers for Grad-CAM
    ratio=0.02,  # Ratio for heatmap overlay
    show_box=True,  # Show bounding boxes on the image
    renormalize=False,  # Do not renormalize the heatmap
)
dataset = load_dataset("marmal88/skin_cancer")
test_split = dataset["test"]

images = []
ground_truths = []
for i in range(9):
    print(f"Image {i}")
    ground_truths.append(test_split[i]["dx"])
    # Generate Grad-CAM heatmaps for an image
    images.append(model(
        img_path=f"test_img/image_{i}.png",  # Path to your input image
    ))

    # Display the results
for image in images:
    display_images(image)
# for i, image_set in enumerate(images):
#     truth = ground_truths[i]
#     if image_set:  # Check if image_set is not empty
#         num_images = len(image_set)
#         fig, axes = plt.subplots(1, num_images, figsize=(15, 5))
#
#         if num_images == 1:
#             axes = [axes]  # Ensure axes is iterable
#
#         titles = ["Heatmap", "Heatmap", "Overlay"]
#         for j, img in enumerate(image_set):
#             axes[j].imshow(img)
#             axes[j].axis('off')
#             axes[j].set_title(titles[j])
#         fig.suptitle(f"Image {i} with label: {truth}", fontsize=16) # Add a super title to the figure.
#         plt.tight_layout()
#         plt.show()
#     else:
#         print(f"Warning: No images generated for image_{i}.png")
