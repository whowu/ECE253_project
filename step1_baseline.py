import pandas as pd
from ultralytics import YOLO
from config import MODEL_PATH, BASELINE_SCENARIOS, TARGET_CLASSES
from utils import create_yaml, plot_metrics
import os

def run_step1():
    print("\n" + "="*40)
    print("STEP 1: Baseline Evaluation")
    print("="*40)
    
    model = YOLO(MODEL_PATH)
    results = []
    
    for scen in BASELINE_SCENARIOS:
        print(f"Evaluating {scen['name']}...")
        yaml_file = create_yaml(scen['path'], scen['split'], "step1")
        
        try:
            metrics = model.val(data=yaml_file, verbose=False)
            for cid, cname in TARGET_CLASSES.items():
                if cid in metrics.box.ap_class_index:
                    idx = list(metrics.box.ap_class_index).index(cid)
                    results.append({
                        "Scenario": scen['name'],
                        "Class": cname,
                        "mAP@50-95": metrics.box.maps[idx]
                    })
        except Exception as e:
            print(f"Error in {scen['name']}: {e}")
        finally:
            if os.path.exists(yaml_file): os.remove(yaml_file)
            
    df = pd.DataFrame(results)
    if not df.empty:
        plot_metrics(df, "Step 1: Baseline vs Distortions", "step1_baseline.png")
    return df

if __name__ == "__main__":
    run_step1()