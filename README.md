# ECE 253 Project - Team RHS

## Overview
**Team RHS** presents this project for **ECE 253: Fundamentals of Digital Image Processing**. 

**Goal:** Evaluate and enhance YOLO11 object detection performance on distorted images (Noise, Motion Blur, Spatial Blur).

**Approaches:**
1.  **Image Restoration:** Pre-processing images to remove distortions before detection.
2.  **Fine-tuning:** Retraining the YOLO model on distorted datasets.

## Project Structure
```text
.
├── algorithms/             # Restoration algorithm implementations
├── config.py               # Global configuration (paths, params)
├── datasets/               # Dataset root directory
├── download_coco.py        # Script to download clean COCO subset (D)
├── main.py                 # Master script for the full pipeline
├── models/                 # YOLO model weights
├── requirements.txt        # Python dependencies
├── results/                # Output metrics and plots
├── runs/                   # YOLO training logs
├── step1_baseline.py       # Step 1: Baseline evaluation
├── step2_optimize.py       # Step 2: Algorithm optimization & processing
├── step3_finetune.py       # Step 3: Fine-tuning comparison
└── utils.py                # Utility functions
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/whowu/ECE253_project.git
    cd ECE253_project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset Setup

You have two options to set up the data: **Download Pre-prepared Data** (Recommended) or **Generate from Scratch**.

### Option 1: Quick Start (Recommended)
For full reproducibility, download our complete dataset package, which includes the clean subset ($D$), distorted versions ($D'$), and pre-processed images (`temp_processed`).

[**📂 Download Complete Dataset (Google Drive)**](https://drive.google.com/drive/folders/1OR3uKI6bM9DtiyBHx6ExYk4VXus16v-e?usp=share_link)

1.  Download and extract the content.
2.  Place folders into `datasets/` so the structure looks like this:
    ```text
    datasets/
    ├── D/                  # Clean COCO validation subset
    ├── D'/                 # Distorted datasets
    └── temp_processed/     # Pre-processed images (ready for Step 3)
    ```

### Option 2: Manual Setup
If you prefer to generate the data yourself:

1.  **Generate Clean Data ($D$):**
    Run the downloader script to fetch images (Person, Backpack, Chair) from COCO 2017:
    ```bash
    python download_coco.py
    ```

2.  **Prepare Distorted Data ($D'$):**
    Ensure your distorted datasets are organized as follows:
    ```text
    datasets/D'/
    ├── motion_blur/
    │   ├── images/
    │   └── labels/
    ├── noise/ ...
    └── spatial_blur/ ...
    ```

3.  **Generate Processed Data (`temp_processed`):**
    > **⚠️ Critical Note:** The image processing logic in `step2_optimize.py` is **commented out by default** to save time on repeated runs.
    >
    > To generate `datasets/temp_processed/` for the first time, you must:
    > 1. Open `step2_optimize.py`.
    > 2. **Uncomment** the code block marked `"Turned off: Datasets Processed"`.
    > 3. Run Step 2 (see Usage).

## Usage

### 1. Run Full Pipeline
Executes Step 1 (Baseline), Step 2 (Optimization), and Step 3 (Fine-tuning) sequentially.
```bash
python main.py
```

### 2. Run Individual Steps

**Step 1: Baseline Evaluation**
Evaluates YOLO11 on clean ($D$) and distorted ($D'$) datasets.
```bash
python step1_baseline.py
```

**Step 2: Algorithm Optimization**
Performs grid search for restoration parameters.
*Remember to uncomment the processing logic if you need to generate processed datasets.*
```bash
python step2_optimize.py
```

**Step 3: Fine-tuning vs. Restoration**
Compares the best restoration method against a fine-tuned model.
*Note: When run standalone, this uses mock config. For real results, run via `main.py`.*
```bash
python step3_finetune.py
```

## Results
Artifacts are saved in `results/`:
*   `step1_baseline_results.csv`
*   `step2_optimization_results.csv`
*   `step3_finetune_results.csv`
*   Performance plots (mAP, Precision, Recall, F1).
