import os
import yaml
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from config import COCO_CLASSES, RESULTS_DIR
from algorithms.denoising import apply_bilateral, apply_cbm3d
from algorithms.motion_blur import process_adaptive, process_wiener
from algorithms.spatial_blur import spatial_contrast, spatial_sharpen

def create_yaml(dataset_path, img_dir, name_suffix=""):
    """Generates a temporary YAML file for YOLO validation/training."""
    yaml_content = {
        "path": os.path.abspath(dataset_path),
        "train": img_dir,  # Often same for simple val
        "val": img_dir,
        "names": COCO_CLASSES,
    }
    filename = f"temp_{name_suffix}.yaml"
    with open(filename, "w") as f:
        yaml.dump(yaml_content, f)
    return filename


def plot_metrics(df, title, filename, group_col="Scenario", metric="mAP@50-95"):
    """Generic plotting function."""
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    # Check if we have data
    if df.empty:
        print(f"⚠️ No data to plot for {filename}")
        return

    ax = sns.barplot(data=df, x="Class", y=metric, hue=group_col, palette="viridis")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)

    save_path = os.path.join(RESULTS_DIR, filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Plot saved: {save_path}")


def apply_algorithm(src_dir, dst_dir, algo_name, param):
    """
    Routes the processing to specific algorithm functions defined in denoising.py.

    Args:
        src_dir: Directory containing distorted images.
        dst_dir: Directory to save processed images.
        algo_name: Name of algo (e.g., 'Bilateral', 'CBM3D').
        param: The parameter(s) for the algorithm.
    """
    # 1. Map string names to actual functions
    # These functions must accept (src_dir, dst_dir, param)
    ALGO_MAP = {
        "Richardson-Lucy": process_adaptive,
        "Wiener": process_wiener,
        "Bilateral": apply_bilateral,
        "CBM3D": apply_cbm3d,
        "CLAHE": spatial_contrast,
        "Spatial_sharpen": spatial_sharpen,
    }

    # 2. Get the function
    processor_func = ALGO_MAP.get(algo_name)

    if not processor_func:
        print(
            f"   ⚠️ Warning: Algorithm '{algo_name}' not implemented in map. Skipping."
        )
        return

    # 3. Clean/Create Destination
    if os.path.exists(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir)

    print(f"   ⚙️ Applying {algo_name} (param={param}) to {src_dir}...")

    # 4. Execute the Algorithm (Directory-Level)
    try:
        processor_func(src_dir, dst_dir, param)
    except Exception as e:
        print(f"   ❌ Critical Error running {algo_name}: {e}")


def split_dataset(src_root, train_ratio=0.8):
    """
    Splits a dataset folder into train/test subfolders.
    Assumes structure: src_root/images and src_root/labels
    """
    images_dir = os.path.join(src_root, "images")
    labels_dir = os.path.join(src_root, "labels")

    # Create new split dirs
    split_root = src_root + "_split"

    if os.path.exists(split_root):
        shutil.rmtree(split_root)

    for split in ["train", "test"]:
        os.makedirs(os.path.join(split_root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(split_root, "labels", split), exist_ok=True)

    all_images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".png"))]
    train_imgs, test_imgs = train_test_split(
        all_images, train_size=train_ratio, random_state=42
    )

    def copy_files(file_list, split_name):
        for img_name in file_list:
            # Copy Image
            shutil.copy(
                os.path.join(images_dir, img_name),
                os.path.join(split_root, "images", split_name, img_name),
            )

            # Copy Label (if exists)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            src_label = os.path.join(labels_dir, label_name)
            if os.path.exists(src_label):
                shutil.copy(
                    src_label,
                    os.path.join(split_root, "labels", split_name, label_name),
                )

    copy_files(train_imgs, "train")
    copy_files(test_imgs, "test")

    return split_root
