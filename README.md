# Brain MRI Tumor Segmentation — Classical + Deep Learning (U-Net)

A Python notebook implementing and comparing four segmentation algorithms on brain MRI scans, with full quantitative evaluation against expert-annotated ground truth masks.

---

## Project Overview

This project segments brain tumors from MRI images using three classical computer vision methods and one deep learning model, then evaluates all four against radiologist-annotated ground truth masks.

**Algorithms implemented:**

| Algorithm | Category |
|---|---|
| Otsu Thresholding | Thresholding-based segmentation |
| K-means Clustering | Clustering-based segmentation |
| Canny Edge Detection + post-processing | Edge-based segmentation |
| U-Net | Deep Learning segmentation |

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

> Each `.tif` image has a corresponding `_mask.tif` file containing the ground truth binary segmentation mask drawn by medical experts.

---

## Project Structure

```
├── brain_mri_segmentation.ipynb   # Main notebook (all code)
├── README.md                       # This file
├── unet_best.pth                        # Saved U-Net checkpoint (generated after training)
├── unet_training_curves.png             # Loss and Dice curves (generated after training)
├── all_algorithms_comparison.png        # Visual overlay comparison (generated after running)
└── metrics_comparison_all.png           # Bar chart and line plot (generated after running)
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
torch
torchvision
```

Install all at once:
```bash
pip install opencv-python scikit-image scikit-learn matplotlib numpy pandas tqdm torch torchvision
```

---

## How to Run

### Option 1: Kaggle (Recommended)

1. Go to the [dataset page on Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
2. Click **New Notebook**
3. Upload `brain_mri_segmentation.ipynb` via **File → Import Notebook**
4. Enable GPU: **Settings → Accelerator → GPU T4 x2**
5. The dataset path is already set correctly — click **Run All**

> GPU is strongly recommended for U-Net training. On CPU, training 30 epochs may take 1–2 hours. On Kaggle GPU, it takes ~10 minutes.

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
| 1 | Import libraries, detect GPU/CPU |
| 2 | Load dataset, filter tumor images, train/val/test split (70/15/15%) |
| 3 | Preprocessing: resize to 256×256, extract FLAIR channel, normalize |
| 4 | Evaluation metrics: Dice, IoU, Precision, Recall, F1, Accuracy |
| 5.1 | Otsu Thresholding implementation |
| 5.2 | K-means Clustering implementation (k=3) |
| 5.3 | Canny Edge Detection + flood fill + morphological post-processing |
| 5.4 | Run classical algorithms on test set |
| 6.1 | PyTorch Dataset class with augmentation (flip, rotation) |
| 6.2 | U-Net architecture (encoder, bottleneck, decoder, skip connections) |
| 6.3 | Combined Dice + BCE loss function |
| 6.4 | Training loop with Adam optimizer, LR scheduler, best checkpoint saving |
| 6.5 | Training curves (loss and Dice over epochs) |
| 6.6 | Evaluate U-Net on test set |
| 7 | Visual comparison of all four algorithms with colored overlays |
| 8 | Bar chart and line plot comparing all metrics |
| 9 | Final results table, best algorithm per metric |
| 10 | Conclusions and discussion |

---

## Preprocessing Details

- **Input channel:** FLAIR (channel index 1 of the 3-channel `.tif` file)
  — Tumors appear hyperintense (brighter) in FLAIR, making this channel ideal for segmentation
- **Resize:** All images standardized to 256×256 pixels
- **Normalization:** Pixel values normalized to [0, 255] for classical algorithms; [0, 1] for U-Net
- **Mask binarization:** Ground truth masks thresholded at 127 → {0, 1}
- **Data split:** Only images containing at least one tumor pixel are used

---

## Algorithm Details

### Otsu Thresholding
Automatically selects the threshold that maximizes inter-class variance between tumor and background. Applied after Gaussian blur (5×5). Post-processed with morphological opening and closing. Final output is the largest connected component.

### K-means Clustering (k=3)
Groups all pixels into 3 clusters by intensity. The cluster with the highest mean intensity is selected as tumor (FLAIR hyperintensity). Same morphological post-processing and connected component filtering as Otsu.

### Canny Edge Detection + Post-processing
Detects tumor boundaries using adaptive thresholds derived from the image median. Edges are dilated to close gaps and flood-filled to create solid regions. A brightness mask (top 25% intensities) removes false positives. Morphological cleanup is applied as a final step.

### U-Net
Encoder-decoder architecture with skip connections. The encoder progressively downsamples the image (256→128→64→32→16) while increasing channels (3→64→128→256→512→1024). The decoder upsamples back to 256×256 using transposed convolutions, with skip connections from corresponding encoder layers to recover spatial detail. Trained end-to-end with combined Dice + BCE loss, Adam optimizer, and ReduceLROnPlateau scheduler.

---

## U-Net Training Configuration

| Parameter | Value |
|---|---|
| Input size | 256×256×3 |
| Encoder features | [64, 128, 256, 512] |
| Bottleneck channels | 1024 |
| Loss function | 0.5 × Dice Loss + 0.5 × BCE Loss |
| Optimizer | Adam (lr=1e-4) |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |
| Epochs | 30 (increase to 50–100 for better results) |
| Batch size | 8 |
| Augmentation | Random horizontal flip, vertical flip, rotation ±15° |
| Checkpoint | Best validation Dice saved to `unet_best.pth` |

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

## Expected Results (approximate)

| Algorithm | Dice | IoU |
|---|---|---|
| Otsu | 0.30–0.45 | 0.20–0.35 |
| K-means | 0.35–0.50 | 0.25–0.40 |
| Canny | 0.20–0.35 | 0.15–0.25 |
| **U-Net** | **0.75–0.88** | **0.60–0.80** |

> Classical algorithms are limited by handcrafted rules. U-Net learns directly from data and significantly outperforms them.

---

## Outputs

After running the full notebook, four files are saved:
- `unet_best.pth` — best U-Net weights (can be reloaded for inference without retraining)
- `unet_training_curves.png` — loss and Dice score over training epochs
- `all_algorithms_comparison.png` — side-by-side visual overlay comparison for 4 samples
- `metrics_comparison_all.png` — bar chart and line plot of average metrics across all algorithms
