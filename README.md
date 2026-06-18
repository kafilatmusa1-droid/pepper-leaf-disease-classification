# Pepper Bell Leaf Disease Classification

COEN807 (Machine Learning for Real-World Data Analytics) term project. This repository contains the complete code, results, and documentation for a supervised image classification system that detects six health-status categories in greenhouse bell pepper leaves: **Bacterial Spot, Cercospora Leaf Spot, Healthy, Leaf Curl, Nutrition Deficiency, and Powdery Mildew**.

## Project Overview

Three convolutional neural network architectures were implemented and compared:

| Model | Approach | Test Accuracy | Macro F1 |
|---|---|---|---|
| Baseline CNN | Trained from scratch | 93.78% | 0.92 |
| **MobileNetV2** | Transfer learning (frozen ImageNet backbone) | **98.50%** | **0.99** |
| ResNet50V2 | Transfer learning (frozen ImageNet backbone) | 98.43% | 0.99 |
| MobileNetV2 (fine-tuned) | Last 20 layers unfrozen | 97.00% | — |

**Final recommended model: MobileNetV2 (frozen)** — highest accuracy, most parameter-efficient (164,742 trainable params), and the fewest agronomically costly misclassifications (Bacterial Spot mistaken for Healthy).

## Dataset

- **Source:** [Pepper Bell Leaf Disease, Mendeley Data](https://data.mendeley.com/datasets/w9drvtxg9g/1) (DOI: 10.17632/w9drvtxg9g.1)
- **License:** CC BY 4.0
- **Size:** 9,283 RGB JPEG images across 6 classes
- **Note:** The dataset is not included in this repository due to size. Download it directly from the link above, or access the mirrored version used for this project on [Kaggle Datasets](https://www.kaggle.com/datasets/kafilatmusa11/pepper-bell-leaf-disease).

## Repository Structure

```
pepper-leaf-disease-classification/
├── notebooks/
│   └── Pepper_Disease_Classification_COEN807.ipynb   # Full training & evaluation notebook
├── results/
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── confusion_matrix_cnn.png
│   ├── confusion_matrix_mobilenet.png
│   └── confusion_matrix_resnet.png
├── report/
│   ├── COEN807_Technical_Report.docx
│   └── COEN807_Presentation.pptx
├── requirements.txt
└── README.md
```

## How to Reproduce These Results

### 1. Environment
This project was developed and run on **Kaggle Notebooks** using a GPU T4 x2 accelerator, with TensorFlow 2.19.0. To reproduce:

1. Create a free [Kaggle](https://www.kaggle.com) account
2. Download the dataset from the [Mendeley link above](https://data.mendeley.com/datasets/w9drvtxg9g/1) and upload it as a Kaggle Dataset, or use the existing [Kaggle mirror](https://www.kaggle.com/datasets/kafilatmusa11/pepper-bell-leaf-disease)
3. Upload `notebooks/Pepper_Disease_Classification_COEN807.ipynb` as a new Kaggle Notebook
4. Attach the dataset via the **Add Input** panel
5. Set the accelerator to **GPU T4 x2** under notebook settings
6. Run all cells in order from top to bottom

### 2. Dependencies
See `requirements.txt`. Core libraries: `tensorflow`, `scikit-learn`, `matplotlib`, `seaborn`, `split-folders`, `numpy`.

```bash
pip install -r requirements.txt
```

### 3. Execution Workflow
The notebook follows this order:

1. **Setup & verification** — confirm GPU availability and dataset path
2. **Exploratory data analysis** — class counts, sample image visualisation, image property inspection
3. **Preprocessing** — stratified 70/15/15 train/val/test split (seed=42), image resizing to 224×224, normalisation, data augmentation, class weight computation
4. **Model 1: Baseline CNN** — built from scratch, trained with class weighting and early stopping
5. **Model 2: MobileNetV2** — transfer learning with frozen ImageNet backbone
6. **Model 3: ResNet50V2** — transfer learning with frozen ImageNet backbone
7. **Hyperparameter tuning** — fine-tuning experiment on MobileNetV2 (last 20 layers unfrozen)
8. **Evaluation** — accuracy, precision, recall, F1-score, and confusion matrices for all models on the held-out test set

All training uses a fixed random seed (42) for the data split, ensuring reproducible results.

## Key Results

Full results, per-class metrics, and discussion are available in the [Technical Report](report/COEN807_Technical_Report.docx). Summary of confusion matrix findings:

- All three models achieved perfect or near-perfect classification on the three smallest classes (Leaf Curl, Nutrition Deficiency, Powdery Mildew) after class weighting was applied.
- The most consequential error type — a truly diseased leaf (Bacterial Spot) misclassified as Healthy — dropped from 37 cases (baseline CNN) to 5 cases (MobileNetV2), an 86% reduction.
- Fine-tuning MobileNetV2's pretrained layers reduced performance rather than improving it, a documented negative result attributed to the relatively small training set size.

## Author

Kafilat Musa — Ahmadu Bello University, Zaria
COEN807: Machine Learning for Real-World Data Analytics

## License

This project is released under the MIT License. The underlying dataset retains its original CC BY 4.0 licence from Mendeley Data.
