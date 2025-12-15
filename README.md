# ECE 253 Project - Team RHS

## Overview
This project is developed by **Team RHS** for **ECE 253 Fundamentals of Digital Image Processing**. The goal is to evaluate and improve the performance of object detection (YOLO11) on distorted images. We investigate two primary approaches:
1. **Image Restoration:** Applying algorithms to remove distortions (Noise, Motion Blur, Spatial Blur) before detection.
2. **Fine-tuning:** Retraining the model on the distorted datasets.

## Project Structure
```text
.
├── algorithms/             # Implementation of restoration algorithms
├── config.py               # Configuration for paths, model parameters, and algorithms
├── datasets/               # Dataset directory (see Datasets section)
├── download_coco.py        # Script to download COCO dataset
├── main.py                 # Main script to execute the project pipeline
├── models/                 # YOLO model weights
├── requirements.txt        # Python dependencies
├── results/                # Output CSVs and plots
├── runs/                   # YOLO training and validation logs
├── step1_baseline.py       # Step 1: Baseline evaluation
├── step2_optimize.py       # Step 2: Algorithm optimization
├── step3_finetune.py       # Step 3: Fine-tuning comparison
└── utils.py                # Helper functions
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/whowu/ECE253_project.git
   cd ECE253_project
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup
This project requires a specific directory structure.

### Generate Clean Data (D)
We use a subset of the COCO 2017 validation set, filtering for the classes: `person`, `backpack`, and `chair`. Run the downloader script:
```bash
python download_coco.py
```
*This utilizes the `fiftyone` library to download and export the images to `datasets/D`.*

### Distorted Data (D')
Ensure your distorted datasets are placed in `datasets/D'` with the following structure:

```text
datasets/D'/
├── motion_blur/
│   ├── images/
│   └── labels/
├── noise/
│   ├── images/
│   └── labels/
└── spatial_blur/
    ├── images/
    └── labels/
```

## Usage
> **Critical: First-Time Setup & Data Generation** By default, the computationally expensive image processing logic is commented out in `step2_optimize.py`. This prevents the pipeline from redundantly re-processing thousands of images on every run.
>
> **Action Required:** If this is your first run, or if you need to regenerate `the datasets/temp_processed/` folder, you must open `step2_optimize.py` and uncomment the code block marked "Turned off: Datasets Processed" before executing the pipeline.

### Running the Full Pipeline (Recommended)
To run the complete workflow (Step 1 → Step 2 → Step 3), execute:
```bash
python main.py
```

### Running Steps Separately
**Step 1: Baseline**

Evaluates the pre-trained YOLO model on the clean dataset (D) and the distorted datasets (D').
```bash
python step1_baseline.py
```

**Step 2: Optimization**

Performs a grid search over parameters for restoration algorithms.
```bash
python step2_optimize.py
```

**Step 3: Fine-tuning**

Compares the performance of the "Best Restoration Algorithm" (from Step 2) against a model "Fine-Tuned" on the distorted data.
```bash
python step3_finetune.py
```
*Note: When run standalone, this script uses mock configuration data for demonstration. For real results, run via `main.py`*

## Results
All artifacts are saved to the results/ directory:
- **CSVs:** `step1_baseline_results.csv`, `step2_optimization_results.csv`, `step3_finetune_results.csv`
- **Plots:** Bar charts comparing mAP, Precision, Recall, and F1-Score across different scenarios and classes.