import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO

# ==========================================
# 1. CONFIGURATION
# ==========================================

# FULL COCO LIST (Required to make indices 24 and 56 valid)
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

# SUBSET MAPPING (Used only for filtering the RESULTS we care about)
CLASS_NAMES = {
    0: 'person',
    24: 'backpack',
    56: 'chair'
}

# Define the 3 scenarios you want to test
SCENARIOS = [
    {
        "name": "Original (D)",
        "dir": "datasets/D", 
        "is_original": False 
    },
    {
        "name": "Motion Blur",
        "dir": "datasets/D'/motion_blur", 
        "is_original": False
    },
    {
        "name": "Noise",
        "dir": "datasets/D'/noise",
        "is_original": False
    }
]

# Use standard YOLO model
MODEL_PATH = "yolo11s.pt" 

# ==========================================
# 2. HELPER: GENERATE YAML FOR D' DATASETS
# ==========================================
def create_temp_yaml(scenario_name, directory):
    """
    Creates a temporary yaml file for datasets.
    """
    # Absolute path to images
    img_dir = os.path.abspath(os.path.join(directory, "images"))
    
    yaml_content = {
        'path': os.path.abspath(directory), 
        'train': 'images', 
        'val': 'images',   
        'names': COCO_80_CLASSES  # <--- FIX: Pass full list so index 56 is valid
    }
    
    yaml_filename = f"{scenario_name.lower().replace(' ', '_')}_temp.yaml"
    
    with open(yaml_filename, 'w') as f:
        yaml.dump(yaml_content, f)
    
    print(f"📄 Created config for {scenario_name}: {yaml_filename}")
    return yaml_filename

# ==========================================
# 3. MAIN EVALUATION LOOP
# ==========================================
def run_evaluation():
    # Load Model
    print(f"⬇️ Loading Model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model. Make sure {MODEL_PATH} exists or download it.")
        return

    results_data = []

    for scenario in SCENARIOS:
        print(f"\n🚀 Evaluating Scenario: {scenario['name']}...")
        
        # Always create a temp YAML to ensure mapping is correct
        yaml_file = create_temp_yaml(scenario['name'], scenario['dir'])

        # Run Validation
        metrics = model.val(data=yaml_file, split='val', verbose=False)
        
        # Extract Class-Specific Metrics
        for class_id, class_name in CLASS_NAMES.items():
            # Safety check: ensure the class ID exists in the results
            # FIX: Use 'ap_class_index' instead of 'ap50_class'
            if class_id in metrics.box.ap_class_index:
                # Find the index of our class_id in the results array
                idx = list(metrics.box.ap_class_index).index(class_id)
                
                map50 = metrics.box.ap50[idx]
                map95 = metrics.box.maps[idx] # FIX: Use 'maps' (array) instead of 'map' (scalar)
                
                results_data.append({
                    "Scenario": scenario['name'],
                    "Class": class_name,
                    "mAP@50": map50,
                    "mAP@50-95": map95
                })
            else:
                print(f"⚠️ Warning: No predictions/labels found for class {class_name} ({class_id}) in {scenario['name']}")
                results_data.append({
                    "Scenario": scenario['name'],
                    "Class": class_name,
                    "mAP@50": 0.0,
                    "mAP@50-95": 0.0
                })
            
        # Cleanup temp yaml
        if os.path.exists(yaml_file):
            os.remove(yaml_file)

    return pd.DataFrame(results_data)

# ==========================================
# 4. PLOTTING RESULTS
# ==========================================
def plot_results(df):
    sns.set_theme(style="whitegrid")
    
    # Create a figure with two subplots (mAP@50 and mAP@50-95)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: mAP@50
    sns.barplot(data=df, x="Class", y="mAP@50", hue="Scenario", ax=axes[0], palette="viridis")
    axes[0].set_title("Comparison of mAP@50 (IoU=0.50)")
    axes[0].set_ylim(0, 1.1)
    
    # Plot 2: mAP@50-95
    sns.barplot(data=df, x="Class", y="mAP@50-95", hue="Scenario", ax=axes[1], palette="viridis")
    axes[1].set_title("Comparison of mAP@50-95 (Robust Metric)")
    axes[1].set_ylim(0, 1.1)
    
    plt.tight_layout()
    save_path = "results/yolo_performance_comparison.png"
    plt.savefig(save_path)
    print(f"\n✅ Comparison plot saved to: {save_path}")
    plt.show()

# ==========================================
# EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Run Evaluation
    df_results = run_evaluation()
    
    if df_results is not None and not df_results.empty:
        os.makedirs("results", exist_ok=True)

        # 2. Print Table
        print("\n📊 Final Results Table:")
        print(df_results)
        
        # 3. Save CSV
        df_results.to_csv("results/evaluation_results.csv", index=False)
        
        # 4. Plot
        plot_results(df_results)
    else:
        print("No results generated.")