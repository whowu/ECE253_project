import pandas as pd
import os
import shutil
from ultralytics import YOLO
from config import MODEL_PATH, ALGO_CONFIGS, DATASET_ROOT, TARGET_CLASSES
from utils import create_yaml, plot_metrics, apply_algorithm


def run_step2():
    print("\n" + "=" * 40)
    print("STEP 2: Algorithm Parameter Search")
    print("=" * 40)

    model = YOLO(MODEL_PATH)
    best_configs = {}  # To store best setting for Step 3

    # Loop over distortions (motion_blur, noise, spatial_blur)
    for distortion, algos in ALGO_CONFIGS.items():
        print(f"\n>>> Analyzing Distortion: {distortion}")
        results = []

        # Define paths
        original_dist_path = os.path.join(DATASET_ROOT, f"D'/{distortion}")
        original_img_dir = "images"  # Assuming images are here

        # 1. Evaluate Original Distorted (Baseline for this plot)
        # Note: In real logic, you might reuse Step 1 result, but we re-run for simplicity
        yaml_base = create_yaml(original_dist_path, original_img_dir, "base")
        m_base = model.val(data=yaml_base, verbose=False, device='mps')
        for cid, cname in TARGET_CLASSES.items():
            if cid in m_base.box.ap_class_index:
                idx = list(m_base.box.ap_class_index).index(cid)
                results.append(
                    {
                        "Setting": "Original",
                        "Class": cname,
                        "mAP@50-95": m_base.box.maps[idx],
                    }
                )
        os.remove(yaml_base)

        # 2. Loop Algos and Params
        for algo_name, params in algos.items():
            for i, p in enumerate(params, 1):
                setting_name = f"{algo_name}_p{i}"

                # Create temporary processed dataset folder
                processed_path = os.path.join(
                    DATASET_ROOT, "temp_processed", distortion, setting_name
                )
                processed_img_dir = "images"

                # Apply Algorithm (Placeholder function)
                # Need absolute path for images
                src_img = os.path.join(original_dist_path, original_img_dir)
                dst_img = os.path.join(processed_path, processed_img_dir)
                
                # ========================= Turned off: Datasets Processed ===========================
                # apply_algorithm(src_img, dst_img, algo_name, p)

                # # Copy labels
                # src_labels = os.path.join(original_dist_path, "labels")
                # dst_labels = os.path.join(processed_path, "labels")

                # if os.path.exists(dst_labels):
                #     shutil.rmtree(dst_labels)

                # if os.path.exists(src_labels):
                #     shutil.copytree(src_labels, dst_labels)
                # else:
                #     print(f"⚠️ Warning: No labels found at {src_labels}")
                # ====================================================================================

                # Evaluate
                yaml_proc = create_yaml(processed_path, processed_img_dir, "proc")
                try:
                    metrics = model.val(data=yaml_proc, verbose=False, device='mps')
                    current_map_sum = 0
                    for cid, cname in TARGET_CLASSES.items():
                        if cid in metrics.box.ap_class_index:
                            idx = list(metrics.box.ap_class_index).index(cid)
                            score = metrics.box.maps[idx]
                            results.append(
                                {
                                    "Setting": setting_name,
                                    "Class": cname,
                                    "mAP@50-95": score,
                                    "raw_algo": algo_name,
                                    "raw_param": p,
                                }
                            )
                            current_map_sum += score
                except Exception as e:
                    print(f"Failed {setting_name}: {e}")
                finally:
                    if os.path.exists(yaml_proc):
                        os.remove(yaml_proc)

        # 3. Plot & Find Winner
        df = pd.DataFrame(results)
        plot_metrics(
            df,
            f"Step 2: Optimization for {distortion}",
            f"step2_{distortion}.png",
            group_col="Setting",
        )

        # Determine "Best" (highest average mAP across 3 classes)
        # Filter out 'Original' for finding best algo
        df_algos = df[df["Setting"] != "Original"]
        if not df_algos.empty:
            best_setting = df_algos.groupby("Setting")["mAP@50-95"].mean().idxmax()
            best_row = df_algos[df_algos["Setting"] == best_setting].iloc[0]
            best_configs[distortion] = {
                "setting": best_setting,
                "algo": best_row["raw_algo"],
                "param": best_row["raw_param"],
            }
            print(f"🏆 Best for {distortion}: {best_setting}")

    return best_configs


if __name__ == "__main__":
    run_step2()
