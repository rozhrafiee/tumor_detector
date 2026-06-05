# Brain MRI Tumor Segmentation — Classical Algorithms

A Python notebook implementing and comparing three classical image segmentation algorithms on brain MRI scans, with full quantitative evaluation.

---

## Project Overview

This project segments brain tumors from MRI images using classical computer vision methods and evaluates each algorithm against radiologist-annotated ground truth masks.

**Algorithms implemented:**
- Otsu Thresholding
- K-means Clustering
- Canny Edge Detection + post-processing

**Evaluation metrics:**
- Dice Coefficient
- IoU (Jaccard Index)
- Precision, Recall, F1-score, Accuracy

---

## Dataset

**Name:** LGG Brain MRI Segmentation  
**Source:** [Kaggle — mateuszbuda/lgg-mri-segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)  
**Size:** ~3,900 brain MRI images from 110 patients  
**Format:** `.tif` files (3-channel: pre-contrast, FLAIR, post-contrast) + binary mask per image  
**Task:** Segment FLAIR abnormalities (lower-grade glioma tumors)

> Each `.tif` image has a corresponding `_mask.tif` file containing the ground truth binary segmentation mask.

---

## Project Structure

```
├── brain_mri_segmentation.ipynb   # Main notebook (all code)
├── README.md                       # This file
├── segmentation_visual_comparison.png   # Generated after running
└── metrics_comparison.png               # Generated after running
```

---

## Requirements

```
python >= 3.8
opencv-python
scikit-image
scikit-learn
matplotlib
numpy
pandas
tqdm
```

Install all at once:
```bash
pip install opencv-python scikit-image scikit-learn matplotlib numpy pandas tqdm
```

---

## How to Run

### Option 1: Kaggle (Recommended)

1. Go to the [dataset page on Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
2. Click **New Notebook**
3. Upload `brain_mri_segmentation.ipynb` via **File → Import Notebook**
4. The dataset path is already set correctly — just click **Run All**

### Option 2: Local (VS Code / Jupyter)

1. Download and unzip the dataset from Kaggle
2. Open the notebook and update this line in Section 2:
   ```python
   DATASET_PATH = 'kaggle_3m'   # path to your unzipped folder
   ```
3. Run all cells:
   ```bash
   jupyter notebook brain_mri_segmentation.ipynb
   ```

---

## Notebook Sections

| Section | Description |
|---|---|
| 1 | Import libraries |
| 2 | Load dataset, filter tumor images, select samples |
| 3 | Preprocessing: resize to 256×256, extract FLAIR channel, normalize |
| 4.1 | Otsu Thresholding implementation |
| 4.2 | K-means Clustering implementation (k=3) |
| 4.3 | Canny Edge Detection + flood fill + morphological post-processing |
| 5 | Evaluation metrics: Dice, IoU, Precision, Recall, F1, Accuracy |
| 6 | Run all algorithms on 50 samples, compute average metrics |
| 7 | Visual comparison: overlay masks on FLAIR images |
| 8 | Bar chart + line plot comparing all metrics |
| 9 | Final results table, best algorithm per metric |
| 10 | Conclusions and discussion |

---

## Preprocessing Details

- **Input channel:** FLAIR (channel index 1 of the 3-channel `.tif` file)
  - Tumors are hyperintense (brighter) in FLAIR sequences, making this channel ideal for segmentation
- **Resize:** All images resized to 256×256 pixels
- **Normalization:** Pixel values normalized to [0, 255]
- **Mask binarization:** Ground truth masks thresholded at 127 → {0, 1}

---

## Algorithm Details

### Otsu Thresholding
Automatically selects the threshold that maximizes inter-class variance between tumor and background pixels. Applied after Gaussian blur (5×5). Post-processed with morphological opening and closing to remove noise and fill holes. Final output is the largest connected component.

### K-means Clustering (k=3)
Groups all pixels into 3 clusters based on intensity. The cluster with the highest mean intensity is assumed to be the tumor (FLAIR hyperintensity). Same morphological post-processing and connected component filtering as Otsu.

### Canny Edge Detection
Detects tumor boundaries using adaptive thresholds derived from the image median. Edges are dilated and flood-filled to create a solid region. A brightness mask (top 25% intensities) is applied to eliminate false positives. Heavy morphological cleanup is then applied.

---

## Evaluation Metric Formulas

| Metric | Formula |
|---|---|
| Dice | 2·TP / (2·TP + FP + FN) |
| IoU | TP / (TP + FP + FN) |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1 | 2 · Precision · Recall / (Precision + Recall) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |

> A small epsilon (1e-8) is added to all denominators to avoid division by zero.

---

## Key Findings

- Tumors appear **hyperintense** in FLAIR MRI, which all three algorithms exploit
- **Otsu** is the fastest but tends to segment all bright regions (skull, ventricles) alongside the tumor
- **K-means** (k=3) better discriminates tumor from other bright regions by splitting intensities into three groups
- **Canny** produces accurate boundary maps but requires extensive post-processing to compute area-based metrics like Dice and IoU
- For significantly better performance, deep learning models (e.g., U-Net) are recommended

---

## Outputs

After running the notebook, two figures are saved:
- `segmentation_visual_comparison.png` — side-by-side visual overlay comparison for 3 samples
- `metrics_comparison.png` — bar chart and line plot of average metrics across all samples
