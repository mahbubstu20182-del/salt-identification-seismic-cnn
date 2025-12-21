# Seismic Salt Identification using U-Net

This repository presents an end-to-end deep learning workflow for **salt body segmentation from seismic images** using the **TGS Salt Identification Challenge dataset**. The project focuses on implementing a robust segmentation pipeline with a **U-Net architecture** and systematically addressing key challenges such as **class imbalance**, limited training data, and model interpretability.

## Problem Background

Salt structures play a critical role in seismic interpretation but are difficult to delineate due to complex subsurface textures and their relatively small spatial extent compared to surrounding sediments. In the TGS dataset, salt pixels represent only a small fraction of the image, making the segmentation task highly imbalanced and challenging for standard convolutional neural networks.

The objective of this project is to perform **pixel-level segmentation of salt bodies** from grayscale seismic images while emphasizing methodological correctness and interpretability.

## Dataset

- **Source:** TGS Salt Identification Challenge (Kaggle)
- **Input:** 101 × 101 grayscale seismic images
- **Output:** Binary salt masks indicating salt and non-salt regions

## Methodology

### Model Architecture
- **U-Net** encoder–decoder architecture with skip connections
- Designed to preserve spatial resolution and boundary information
- Implemented using **PyTorch**

### Training Strategy
- Image preprocessing: normalization and resizing
- Mini-batch training using multiple image–mask pairs
- Optimizer: Adam
- Loss functions:
  - Binary Cross-Entropy (BCE)
  - Dice Loss to mitigate severe class imbalance between salt and background pixels

### Evaluation
- Model predictions converted to probability maps using sigmoid activation
- Binary masks generated through thresholding
- Qualitative evaluation performed using transparent overlay visualization on seismic images
- 
## Results and Observations

Initial experiments using BCE loss resulted in background-dominated predictions due to the strong class imbalance inherent in the dataset. Introducing Dice loss improved sensitivity to thin and sparse salt structures, leading to more localized and interpretable predictions. Multi-image training further stabilized model behavior and reduced extreme over- or under-prediction.

The results demonstrate correct pipeline implementation and highlight common challenges in seismic image segmentation, particularly the trade-off between sensitivity and specificity in imbalanced datasets.

## How to Run

1. Install dependencies:
   pip install -r requirements.txt
2. Open the notebook:
notebooks/tgs_unet_salt_segmentation.ipynb
3. Run all cells sequentially (recommended in Google Colab with GPU support).
Limitations and Future Work

Training was performed on a subset of the dataset due to computational constraints

## Further improvements are expected by:
Training on the full dataset (4000+ images)
Increasing the number of training epochs
Applying data augmentation
Exploring advanced loss functions such as weighted Dice or focal loss
Quantitative evaluation using Dice coefficient and IoU metrics

## Author
Mahbub Alam
MSc in Geological Sciences (Remote Sensing & Environmental Sciences)
Jahangirnagar University
