import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================
COCO_80_CLASSES = [
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

CLASS_NAMES = {
    0: 'person',
    24: 'backpack',
    56: 'chair'
}

SCENARIOS = [
    {
        "name": "Original (D)",
        "dir": "datasets/D",
        "val_subdir": "images/val",
        "is_original": False 
    },
    {
        "name": "Motion Blur",
        "dir": "datasets/D'/motion_blur",
        "val_subdir": "images",
        "is_original": False
    },
    {
        "name": "Noise",
        "dir": "datasets/D'/noise",
        "val_subdir": "images",
        "is_original": False
    }
]

MODEL_PATH = "models/yolo11s.pt" 

# ==========================================
# 2. HELPER
# ==========================================
def create_temp_yaml(scenario, directory):
    val_path = scenario['val_subdir']
    yaml_content = {
        'path': os.path.abspath(directory), 
        'train': val_path, 
        'val': val_path,   
        'names': COCO_80_CLASSES 
    }
    yaml_filename = f"{scenario['name'].lower().replace(' ', '_')}_temp.yaml"

    with open(yaml_filename, 'w') as f:
        yaml.dump(yaml_content, f)

    return yaml_filename

# ==========================================
# 3. MAIN EVALUATION LOOP (UPDATED)
# ==========================================
def run_evaluation():
    print(f"⬇️ Loading Model: {MODEL_PATH}")

    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return pd.DataFrame()

    results_data = []

    for scenario in SCENARIOS:
        print(f"\n🚀 Evaluating Scenario: {scenario['name']}...")
        yaml_file = create_temp_yaml(scenario, scenario['dir'])

        try:
            # Run Validation
            metrics = model.val(data=yaml_file, split='val', verbose=False)
            
            for class_id, class_name in CLASS_NAMES.items():
                if class_id in metrics.box.ap_class_index:
                    idx = list(metrics.box.ap_class_index).index(class_id)
                    
                    # === NEW METRICS EXTRACTED HERE ===
                    map50 = metrics.box.ap50[idx]
                    map95 = metrics.box.maps[idx]
                    precision = metrics.box.p[idx] # Precision at best confidence
                    recall = metrics.box.r[idx]    # Recall at best confidence
                    f1 = metrics.box.f1[idx]       # F1 Score
                    
                    results_data.append({
                        "Scenario": scenario['name'],
                        "Class": class_name,
                        "mAP@50": map50,
                        "mAP@50-95": map95,
                        "Precision": precision,
                        "Recall": recall,
                        "F1-Score": f1
                    })
                else:
                    results_data.append({
                        "Scenario": scenario['name'],
                        "Class": class_name,
                        "mAP@50": 0.0, "mAP@50-95": 0.0, 
                        "Precision": 0.0, "Recall": 0.0, "F1-Score": 0.0
                    })
        except Exception as e:
            print(f"❌ Error evaluating {scenario['name']}: {e}")
        finally:
            if os.path.exists(yaml_file):
                os.remove(yaml_file)

    return pd.DataFrame(results_data)

# ==========================================
# 4. PLOTTING RESULTS (UPDATED 2x2 GRID)
# ==========================================
def plot_results(df):
    sns.set_theme(style="whitegrid")
    
    # 2x2 Subplots to fit all metrics
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. mAP@50-95 (The "Gold Standard" metric)
    sns.barplot(data=df, x="Class", y="mAP@50-95", hue="Scenario", ax=axes[0,0], palette="viridis")
    axes[0,0].set_title("mAP@50-95 (Overall Quality)")
    axes[0,0].set_ylim(0, 1.1)

    # 2. F1-Score (Balance of P & R)
    sns.barplot(data=df, x="Class", y="F1-Score", hue="Scenario", ax=axes[0,1], palette="viridis")
    axes[0,1].set_title("F1-Score (Harmonic Mean)")
    axes[0,1].set_ylim(0, 1.1)

    # 3. Precision (False Positives)
    sns.barplot(data=df, x="Class", y="Precision", hue="Scenario", ax=axes[1,0], palette="viridis")
    axes[1,0].set_title("Precision (Avoids Hallucinations)")
    axes[1,0].set_ylim(0, 1.1)

    # 4. Recall (False Negatives)
    sns.barplot(data=df, x="Class", y="Recall", hue="Scenario", ax=axes[1,1], palette="viridis")
    axes[1,1].set_title("Recall (Finds All Objects)")
    axes[1,1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save to results folder
    save_path = "results/yolo_performance_comparison.png"
    plt.savefig(save_path)
    print(f"\n✅ Comparison plot saved to: {save_path}")

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    df_results = run_evaluation()

    if df_results is not None and not df_results.empty:
        os.makedirs("results", exist_ok=True)

        # Print Table
        print("\n📊 Final Results Table:")
        print(df_results)

        # Save CSV
        df_results.to_csv("results/evaluation_results.csv", index=False)

        # Plot
        plot_results(df_results)
    else:
        print("No results generated.")