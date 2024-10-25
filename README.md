# 🧠 Brain Tumor Segmentation with U-Net 🧠

Welcome to the **Brain Tumor Segmentation** project! This repository contains a Jupyter Notebook for training and visualizing a U-Net model to perform precise segmentation of brain tumors from medical images. With this tool, you can apply deep learning to identify and segment regions of interest in MRI scans, making it a valuable asset for medical image analysis. 

## 📂 Project Files

- **`Brain_Tumour_Segmentation.ipynb`**: This notebook covers all the steps required to preprocess data, train a U-Net model, and visualize the segmentation results.
- **Model**: The model used here is a U-Net, a convolutional neural network (CNN) architecture designed for image segmentation tasks.

## ✨ Key Features

- **Data Loading and Preprocessing**: Load and preprocess MRI images and masks.
- **Model Training**: Train a U-Net model to identify tumor regions.
- **Prediction and Visualization**: Predict masks for input images and visualize the results alongside the original images and ground truth masks.
- **Inference Timing**: Time the predictions for each image to analyze model performance.

## 📖 About U-Net

The **U-Net** architecture is a popular choice for image segmentation tasks, particularly in biomedical fields. It consists of:
- **Encoder** (Contracting Path): Captures context with convolutional and pooling layers.
- **Decoder** (Expanding Path): Upsamples the feature maps to restore spatial resolution.
- **Skip Connections**: Connect encoder and decoder layers at each level, which helps retain fine-grained spatial information.

### Why U-Net?
U-Net’s design makes it ideal for segmentation tasks, as it combines local and global information through its unique structure. By preserving spatial details in the skip connections, U-Net can create detailed and accurate masks.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python installed, along with the following libraries:
- `numpy`
- `tensorflow` or `keras`
- `matplotlib`
- `opencv-python`

Install them via:
```bash
pip install numpy tensorflow matplotlib opencv-python
```

### Running the Notebook

1. Clone the repository.
2. Open `Brain_Tumour_Segmentation.ipynb` in Jupyter Notebook or JupyterLab.
3. Run each cell step-by-step to:
   - Load and preprocess the data.
   - Train the U-Net model on the brain tumor dataset.
   - Visualize predictions on test images.

### Example Usage

```python
# Predict and visualize segmentation for a sample image
predicted_mask, inference_time = predict_single_image(sample_image, unet_model)
visualize_segmentation_with_cropping(sample_image, predicted_mask, ground_truth_mask)
```

## 📊 Visualization

The notebook includes functions to visualize the original MRI image, ground truth mask, predicted mask, and a segmented version of the image:
- **Original Image**: The MRI image used for prediction.
- **Predicted Mask**: Output from the U-Net model, showing detected tumor areas.
- **Ground Truth Mask**: The actual tumor area.
- **Segmented Image**: The original image with the predicted mask applied, highlighting the tumor region.

This allows easy visual inspection of the model's performance!

## 🤖 Model Evaluation

The notebook also times each prediction, so you can evaluate the model's inference time. This is useful when assessing the model's efficiency and potential for real-time applications.

## 📝 Notes

- Ensure MRI images and ground truth masks are preprocessed to match the input requirements of the model.
- Modify model parameters, batch size, and learning rate to optimize training.

## 📬 Contact

If you have questions, feel free to reach out! We appreciate any feedback or contributions that could improve this project. 

Enjoy segmenting brain tumors with U-Net! 🎉
