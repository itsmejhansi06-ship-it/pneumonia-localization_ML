# Weakly Supervised Pneumonia Localization

## 1. Project Overview

This project focuses on detecting pneumonia from chest X-ray images and approximately localizing the affected regions using deep learning and image-processing techniques.

The project implements a baseline Convolutional Neural Network (CNN), U-Net lung segmentation, a CNN + U-Net model, Class Activation Mapping (CAM), and K-means clustering for localization.

## 2. Objectives

- Classify chest X-ray images as pneumonia or no pneumonia.
- Generate CAM heatmaps to identify important image regions.
- Use U-Net to segment lung regions.
- Compare baseline CNN and CNN + U-Net models.
- Experiment with CAM thresholds and K-means clustering for pneumonia localization.

## 3. Dataset

The project uses the [RSNA Pneumonia Detection dataset](https://www.kaggle.com/datasets/sovitrath/rsna-pneumonia-detection-2018).

The original label CSV contains 26,684 patient records. However, only 4,307 corresponding JPEG images were available locally and used in this project.

| Dataset details | Number of images |
|---|---:|
| Total usable images | 4,307 |
| No pneumonia | 2,767 |
| Pneumonia | 1,540 |
| Training set | 3,014 |
| Validation set | 862 |
| Test set | 431 |

The dataset was split into training, validation, and test sets in an approximate ratio of 70:20:10.

## 4. Methodology

### Baseline CNN

A CNN with convolutional layers, ReLU activation, pooling, global average pooling, and a fully connected layer was trained to classify chest X-ray images.

### U-Net Lung Segmentation

A separate U-Net model was trained using 1,000 lung image-mask pairs. It was used to generate lung masks for the available chest X-ray images.

### CNN + U-Net

The original chest X-ray and its lung-segmented version were combined as a two-channel input to a CNN to investigate whether lung-region information improves classification and localization.

### CAM Localization

Class Activation Mapping generates heatmaps that highlight image regions influencing the model's prediction. These heatmaps are converted into approximate bounding boxes.

### K-means Clustering

K-means clustering was applied to high-activation regions in CAM heatmaps. Different values of K were tested to compare localization performance.

## 5. Results

### Classification Results

| Metric | Baseline CNN | CNN + U-Net |
|---|---:|---:|
| Accuracy | 75.64% | 74.71% |
| Precision | 74.26% | — |
| Recall | 48.70% | — |
| F1-score | 58.82% | — |

### Localization Results

| Metric | Baseline CNN | CNN + U-Net |
|---|---:|---:|
| Mean IoU at default CAM threshold | 0.1001 | 0.1045 |
| Median IoU at default CAM threshold | 0.0882 | 0.0362 |
| Best tested CAM threshold | 0.40 | 0.30 |

For baseline CNN K-means localization, K = 2 achieved a mean IoU of approximately 0.1000.

The U-Net achieved a validation IoU of 0.8761 for lung segmentation. This is a separate task from pneumonia localization.

**Note:** The CAM threshold results are exploratory. The best threshold should ideally be selected using validation data.

## 6. Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- OpenCV
- Scikit-learn
- Matplotlib
- Convolutional Neural Networks (CNN)
- U-Net
- Class Activation Mapping (CAM)
- K-means clustering

## 7. Project Structure

```text
pneumonia-localization_ML/
├── data/
│   ├── processed/
│   └── unet/
├── models/
├── results/
├── src/
│   ├── config.py
│   ├── prepare_dataset.py
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── train_unet.py
│   ├── apply_unet.py
│   ├── train_cnn_unet.py
│   ├── cam.py
│   ├── cam_unet.py
│   ├── threshold_experiment.py
│   ├── cam_threshold_unet.py
│   ├── kmeans_localization.py
│   └── kmeans_unet.py
├── requirements.txt
└── README.md
```

*This is a simplified overview; your actual folders and filenames may differ.*

## 8. How to Run

### Step 1: Clone the repository

```bash
git clone <your-github-repository-url>
cd pneumonia-localization_ML
```

### Step 2: Create and activate a virtual environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the dataset

Download the RSNA dataset and place the images and label CSV in the locations configured in `config.py`.

Run:

```bash
python -m src.prepare_dataset
```

### Step 5: Train and evaluate the baseline CNN

```bash
python -m src.train_classifier --epochs 20
python -m src.evaluate_classifier
```

Other scripts can be run to train U-Net, apply lung segmentation, train CNN + U-Net, and evaluate CAM and K-means localization. Check the script arguments and paths in `config.py` before running them.

## 9. Conclusion

This project demonstrates a weakly supervised approach to pneumonia classification and approximate localization using chest X-ray images. CNN was used for classification, U-Net for lung segmentation, CAM for visual explanation, and K-means for grouping important image regions.

The baseline CNN achieved slightly higher classification accuracy than CNN + U-Net, while the combined model showed a small improvement in mean localization IoU at the reported default threshold. Further work is needed to improve localization accuracy and evaluate the methods using independently selected settings.

