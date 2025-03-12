from YOLOv8_Explainer import yolov8_heatmap, display_images

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
for i in range(10):
    # Generate Grad-CAM heatmaps for an image
    images = model(
        img_path=f"test_img/image_{i}.png",  # Path to your input image
    )

    # Display the results
    display_images(images)