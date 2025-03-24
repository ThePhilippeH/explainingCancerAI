# Explainable AI in Skin Cancer Detection: Comparing CNNs, ViT, and SWIN Transformers

## Introduction
Artificial Intelligence (AI) has revolutionized the field of computer vision, enabling breakthroughs in areas like medical imaging, autonomous vehicles, and more. However, as AI models become more complex, understanding how they make decisions has become increasingly challenging. This is where **AI explainability** comes in—a critical area of research that aims to make AI models more transparent and interpretable.

In this blog post, we explore the explainability of three prominent deep learning architectures **CNNs (YOLOv8)**, **Vision Transformers (ViT)**, and **SWIN Transformers**—in the context of skin cancer detection. Using explainability techniques like **LIME** and **GRAD-CAM**, we compare how well these models can justify their predictions. Whether you're a researcher, developer, or simply curious about AI, this post will provide insights into the trade-offs between model performance and interpretability.

[//]: # (TODO: ADD LINKS TO OG PAPERS)
[//]: # ()
## Background

### What is AI Explainability?
AI explainability refers to the ability of an AI model to provide clear, understandable reasons for its predictions. In certain applications with
In fields like healthcare, explainability is essential for fostering trust among doctors and medical professionals. By providing clear and interpretable insights into how AI models make decisions, explainability helps clinicians understand and validate the reasoning behind predictions. This trust is critical for the successful integration of AI into clinical workflows.

Moreover, AI models have the potential to uncover subtle patterns in medical data that may not be immediately visible to even the most experienced doctors. In the example that we use, in skin cancer detection, AI could identify previously unseen features in dermatological images that correlate with specific conditions. Explainability techniques like LIME and GRAD-CAM allow doctors to visualize these patterns, possibly even bringing to light features that were previously ignored. This would both allow to improve diagnostic accuracy and aid doctors in making more informed decisions.

### The Role of Computer Vision in Medical Imaging
In medical imaging, computer vision models are used to detect diseases, classify conditions, and assist in diagnosis. Skin cancer detection, in particular, has seen significant advancements thanks to deep learning models that can analyze dermatological images with high accuracy.

----
### Models Compared
In this project, we compare three popular deep learning architectures:

#### 1. CNNs (YOLOv8)

Convolutional Neural Networks (CNNs) form the backbone of YOLOv8 by extracting spatial and hierarchical features from images. The core components of CNNs used in YOLOv8 include:

- **Convolutional Layers** – Apply filters to detect edges, textures, and patterns at different levels.
- **Activation Functions** – Use non-linearity (ReLU) to enhance feature learning.
- **Pooling Layers** – Reduce dimensionality while retaining important features.
- **Batch Normalization** – Normalizes activations to stabilize training and improve convergence.
- **Residual Connections** – Improve gradient flow, making training more efficient.



YOLOv8 follows a modular design consisting of three main components:  

##### **a. Backbone (Feature Extraction)**  
The backbone is responsible for extracting useful features from the input image. YOLOv8 uses:  
- **CSPDarkNet53** – A deep CNN with CSP connections to reduce redundancy and enhance efficiency.  
- **C2f (CSP2-Factorized)** – Optimizes feature reuse and improves training stability.  
- **Spatial Pyramid Pooling-Fast (SPPF)** – Captures multi-scale features with minimal computational cost.  

##### **b. Neck (Feature Fusion)**  
The neck aggregates and refines feature maps from different layers for better object detection. It consists of:  
- **Path Aggregation Network (PANet)** – Enhances the flow of spatial and semantic information across layers.  
- **BiFPN (Bi-directional Feature Pyramid Network)** – Efficiently merges low- and high-level features, improving small object detection.  

##### **c. Head (Prediction & Decoding)**  
The prediction head is responsible for detecting objects in the image. Key improvements in YOLOv8 include:  
- **Anchor-Free Detection** – Eliminates predefined anchor boxes, making detection faster and more flexible.  
- **Decoupled Detection Head** – Separates classification and localization tasks, leading to better accuracy.  

![alt text](./images_report//image_yolov8_architecture.png)




#### 1. Vision Transformers (ViTs)  

Vision Transformers (ViTs) are a deep learning architecture designed for image recognition tasks, leveraging the **self-attention mechanism** instead of convolutions to process visual data. Unlike CNNs, which extract features using spatial hierarchies, ViTs **divide an image into fixed-size patches** and process them as a sequence of tokens.  

The core components of ViTs include:  

- **Patch Embedding Layer** – Splits an image into small patches and embeds them into vector representations.  
- **Position Embeddings** – Adds spatial information to maintain positional relationships between patches.  
- **Multi-Head Self-Attention (MHSA)** – Captures global dependencies between image regions.  
- **Feedforward Network (FFN)** – Applies transformations and non-linearities to enhance feature learning.  
- **Layer Normalization & Residual Connections** – Stabilizes training and improves gradient flow.  

ViTs follow a modular design consisting of three main components:  

##### **a. Patch Embedding & Tokenization**  
Instead of using convolutions, ViTs split an image into **fixed-size patches** and flatten them into 1D sequences. Key steps include:  
- **Linear Projection** – Each patch is projected into a high-dimensional embedding space.  
- **Class Token** – A special learnable token is added to the sequence to represent the entire image.  
- **Positional Encoding** – Injects spatial information into the model since self-attention lacks inherent locality.  

##### **b. Transformer Encoder (Feature Extraction)**  
The core of ViTs is the **stacked Transformer encoder**, inspired by NLP models like BERT. Each encoder block contains:  
- **Multi-Head Self-Attention (MHSA)** – Enables the model to capture long-range dependencies between image patches.  
- **Feedforward Network (FFN)** – Applies non-linearity and transformations to enhance learned representations.  
- **Layer Normalization & Residual Connections** – Helps stabilize gradients and improve optimization.  

##### **c. Classification & Output Head**  
After processing through multiple transformer blocks, the model uses the **class token** for final prediction. This step involves:  
- **MLP Head** – A simple fully connected layer that maps the class token representation to output categories.  
- **Softmax Activation** – Converts logits into class probabilities.  

![alt text](./images_report/vit_architecture.png)

#### 1. Swin Transformers  

Swin Transformers (Shifted Window Transformers) are an advanced vision transformer architecture designed to improve computational efficiency and scalability. Unlike standard Vision Transformers (ViTs), which apply **global self-attention** across the entire image, Swin Transformers introduce **hierarchical feature maps** and **shifted window attention**, making them more suitable for dense vision tasks like object detection and segmentation.  

The core components of Swin Transformers include:  

- **Patch Embedding Layer** – Splits an image into smaller non-overlapping patches and embeds them into vectors.  
- **Hierarchical Feature Maps** – Reduces computational complexity by progressively merging patches into coarser representations.  
- **Shifted Window Multi-Head Self-Attention (SW-MHSA)** – Applies self-attention within **local** windows while allowing cross-window interactions.  
- **Feedforward Network (FFN)** – Enhances feature transformations using non-linearity.  
- **Layer Normalization & Residual Connections** – Stabilizes training and ensures better gradient flow.  

Swin Transformers follow a modular design consisting of three main components:  

##### **a. Patch Embedding & Hierarchical Feature Learning**  
Instead of processing an entire image as a single sequence, Swin Transformers organize patches into a **hierarchical structure**.  
- **Linear Projection** – Converts non-overlapping patches into token embeddings.  
- **Patch Merging** – Gradually merges adjacent patches to reduce sequence length, creating multi-scale feature maps.  
- **Positional Encoding (Relative Position Bias)** – Introduces spatial awareness without requiring fixed positional embeddings.  

##### **b. Swin Transformer Blocks (Feature Extraction)**  
The Swin Transformer stack replaces traditional **global self-attention** with **Shifted Window Self-Attention (SW-MHSA)** to enhance efficiency.  
- **Window-Based Multi-Head Self-Attention (W-MHSA)** – Computes self-attention within non-overlapping local windows to reduce complexity.  
- **Shifted Window Mechanism** – Introduces a **window-shifting step** in alternating layers, allowing information exchange between neighboring windows.  
- **Feedforward Network (FFN)** – Applies MLP layers for feature transformation.  
- **Layer Normalization & Residual Connections** – Stabilizes learning and improves model convergence.  

##### **c. Classification & Output Head**  
After processing through Swin Transformer blocks, the model generates hierarchical feature representations that are used for classification or other vision tasks.  
- **Global Average Pooling (GAP)** – Pools feature maps for classification.  
- **Fully Connected Layer (MLP Head)** – Maps the final representation to output categories.  
- **Softmax Activation** – Converts logits into probabilities for classification tasks.  


![alt text](./images_report/swin_architecture.png)

---

## Explainability Techniques
#### LIME
**LIME (Local Interpretable Model-Agnostic Explanations)** is a popular technique designed to **explain the predictions** of any machine learning model by approximating its behavior with an interpretable surrogate model in a local region. The steps of how it works precisely are given in the following list:

1. **Selecting an Instance for Explanation**
     - LIME explains **one prediction at a time**. Given an input instance \( x \), the goal is to understand why the model predicted a certain output \( f(x) \).
  
  2. **Generating Perturbations**
     - LIME creates **slightly modified versions** of \( x \) by randomly perturbing its feature values. This generates a dataset of perturbed samples \( \{ x_1, x_2, ..., x_n \} \).
  
  3. **Getting Predictions from the Black-Box Model**
     - The original model \( f(x) \) is used to predict outcomes for the perturbed instances, producing predictions \( \{ f(x_1), f(x_2), ..., f(x_n) \} \).
  
  4. **Weighting Perturbed Instances**
     - LIME assigns higher importance to perturbed samples that are **closer** to the original instance \( x \) using a similarity function (e.g., exponential kernel).
  
  5. **Training a Simple Surrogate Model**
     - LIME trains an **interpretable model** (e.g., linear regression, decision tree) on the weighted dataset. This model approximates the local decision boundary of the black-box model.
  
  6. **Generating Explanations**
     - The learned interpretable model is analyzed to understand the contribution of each feature towards the prediction. The output is a **feature importance ranking**, showing which features most influenced the decision.



#### Grad-CAM
**Grad-CAM (Gradient-weighted Class Activation Mapping)** is a technique used to generate visual explanations for the predictions made by Convolutional Neural Networks (CNNs). It highlights the regions of the input image that are most important for the model's decision-making process by leveraging gradients. The steps of how it works precisely are given in the following list:

1. **Selecting the Target Class for Explanation**
   - Grad-CAM explains **which parts of the image contributed most** to a specific class prediction. The target class \( c \) (e.g., a particular object in an image) is selected for visualization.
  
2. **Computing Gradients for the Target Class**
   - Grad-CAM computes the **gradients of the target class score** with respect to the output feature maps of the last convolutional layer in the network. These gradients indicate how important each feature map is for the final decision.
  
3. **Global Average Pooling of Gradients**
   - The gradients are then **pooled globally** (via global average pooling) to obtain a weight for each channel of the feature map. These weights represent the importance of each feature map channel in the decision-making process.
  
4. **Weighted Combination of Feature Maps**
   - The feature maps of the last convolutional layer are **weighted by the computed gradients**, and a weighted sum of these feature maps is created. This provides a class-specific saliency map that highlights the regions important for the target class.

5. **Generating the Class Activation Map**
   - The weighted feature maps are passed through a **ReLU activation function** to obtain the final **Class Activation Map (CAM)**. This CAM is a heatmap that shows which parts of the image had the greatest influence on the model’s decision.

6. **Visualizing the CAM on the Input Image**
   - The **Class Activation Map (CAM)** is superimposed onto the original image to visualize the **regions of the image** that contributed most to the target class prediction. High-intensity regions on the heatmap indicate the most influential parts of the image for that class.

---

## Methodology
In the next section, we’ll dive into the methodology, including the datasets used, how the models were trained, and how LIME and GRAD-CAM were applied to evaluate explainability. 

---

## Results and Insights
(To be added after completing the analysis.)

---

## Conclusion
(To be added after completing the analysis.)

---

## References
- [LIME Paper](https://arxiv.org/abs/1602.04938)
- [GRAD-CAM Paper](https://arxiv.org/abs/1610.02391)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ViT Paper](https://arxiv.org/abs/2010.11929)
- [SWIN Transformer Paper](https://arxiv.org/abs/2103.14030)