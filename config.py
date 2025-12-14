import os

# =========================
# GLOBAL SETTINGS
# =========================
# CHANGE THIS to 'cuda' (NVIDIA), 'mps' (Mac), or 'cpu'
DEVICE = "mps"

MODEL_PATH = "models/yolo11s.pt"
RESULTS_DIR = "results"
DATASET_ROOT = "datasets"

# Specific classes you are tracking (from your main.py)
TARGET_CLASSES = {
    0: 'person',
    24: 'backpack',
    56: 'chair'
}

# The 80 COCO classes list (Required for YAML generation)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# =========================
# DATASET DEFINITIONS
# =========================
# Defined based on your description
DISTORTIONS = ['motion_blur', 'noise', 'spatial_blur']

# Step 1 Scenarios
BASELINE_SCENARIOS = [
    {"name": "COCO (D)", "path": os.path.join(DATASET_ROOT, "D"), "split": "images/val"},
    {"name": "Motion Blur (D')", "path": os.path.join(DATASET_ROOT, "D'/motion_blur"), "split": "images"},
    {"name": "Noise (D')", "path": os.path.join(DATASET_ROOT, "D'/noise"), "split": "images"},
    {"name": "Spatial Blur (D')", "path": os.path.join(DATASET_ROOT, "D'/spatial_blur"), "split": "images"},
]

# =========================
# ALGORITHM PARAMETERS (Step 2)
# =========================
# 6 Unique Algorithms (2 per distortion) with 3 tunable params each
ALGO_CONFIGS = {
    'noise': {
        'Bilateral': [(5, 50, 50), (9, 75, 75), (15, 100, 100)],
        'CBM3D': [0.0588, 0.0980, 0.1569]
    },
    'motion_blur': {
        'Richardson-Lucy': [[9, 15, 21, 6, 15, 22], [9, 15, 25, 6, 15, 30], [7, 13, 27, 4, 12, 32]],
        'Wiener': [[9, 15, 21, 0.01], [9, 15, 21, 0.05], [9, 15, 25, 0.02]]
    },
    'spatial_blur': {
        'CLAHE': [[2.0], [2.5], [3.0]],
        'Spatial_sharpen': [[0.8, 1.0], [1.0, 1.2], [1.2, 1.5]]
    }
}