import pandas as pd
import os
import shutil
from ultralytics import YOLO
from config import DEVICE, MODEL_PATH, ALGO_CONFIGS, DATASET_ROOT, TARGET_CLASSES
from utils import clean_yolo_cache, create_yaml, save_results_to_csv, plot_metrics, apply_algorithm


def run_step2():
    print("\n" + "=" * 40)
    print("STEP 2: Algorithm Parameter Search")
    print("=" * 40)

    model = YOLO(MODEL_PATH)
    best_configs = {}  # To store best setting for Step 3
    all_results = []

    # Loop over distortions (noise, motion_blur, spatial_blur)
    for distortion, algos in ALGO_CONFIGS.items():
        print(f"\n>>> Analyzing Distortion: {distortion}")
        current_distortion_results = []

        # Define paths
        original_dist_path = os.path.join(DATASET_ROOT, f"D'/{distortion}")
        original_img_dir = "images"  # Assuming images are here

        # 1. Evaluate Original Distorted (Baseline for this plot)
        # Note: In real logic, you might reuse Step 1 result, but we re-run for simplicity
        clean_yolo_cache(original_dist_path)

        yaml_base = create_yaml(original_dist_path, original_img_dir, "base")
        m_base = model.val(data=yaml_base, verbose=False, device=DEVICE)
        for cid, cname in TARGET_CLASSES.items():
            if cid in m_base.box.ap_class_index:
                idx = list(m_base.box.ap_class_index).index(cid)
                entry = {
                    "Distortion": distortion,
                    "Setting": "Original",
                    "Class": cname,
                    "mAP@50-95": m_base.box.maps[idx],
                    "Precision": m_base.box.p[idx],
                    "Recall": m_base.box.r[idx],
                    "F1-Score": m_base.box.f1[idx],
                    "raw_algo": "None",
                    "raw_param": "None",
                }
                current_distortion_results.append(entry)
                all_results.append(entry)
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

                # ========================= Turned off: Datasets Processed ===========================
                # # Apply Algorithm (Placeholder function)
                # # Need absolute path for images
                # src_img = os.path.join(original_dist_path, original_img_dir)
                # dst_img = os.path.join(processed_path, processed_img_dir)

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

                clean_yolo_cache(processed_path)

                # Evaluate
                yaml_proc = create_yaml(processed_path, processed_img_dir, "proc")
                try:
                    metrics = model.val(data=yaml_proc, verbose=False, device=DEVICE)

                    for cid, cname in TARGET_CLASSES.items():
                        if cid in metrics.box.ap_class_index:
                            idx = list(metrics.box.ap_class_index).index(cid)
                            entry = {
                                "Distortion": distortion,
                                "Setting": setting_name,
                                "Class": cname,
                                "mAP@50-95": metrics.box.maps[idx],
                                "Precision": metrics.box.p[idx],
                                "Recall": metrics.box.r[idx],
                                "F1-Score": metrics.box.f1[idx],
                                "raw_algo": algo_name,
                                "raw_param": p,
                            }
                            current_distortion_results.append(entry)
                            all_results.append(entry)
                except Exception as e:
                    print(f"Failed {setting_name}: {e}")
                finally:
                    if os.path.exists(yaml_proc):
                        os.remove(yaml_proc)

        # 3. Plot & Find Winner
        df_curr = pd.DataFrame(current_distortion_results)

        plot_metrics(
            df_curr,
            f"Step 2: Optimization for {distortion}",
            f"step2_{distortion}",
            group_col="Setting",
        )

        # Determine "Best" (highest average mAP across 3 classes)
        # Filter out 'Original' for finding best algo
        df_algos = df_curr[df_curr["Setting"] != "Original"]
        if not df_algos.empty:
            best_setting = df_algos.groupby("Setting")["mAP@50-95"].mean().idxmax()
            best_row = df_algos[df_algos["Setting"] == best_setting].iloc[0]
            best_configs[distortion] = {
                "setting": best_setting,
                "algo": best_row["raw_algo"],
                "param": best_row["raw_param"],
            }
            print(f"🏆 Best for {distortion}: {best_setting}")

    # Save Unified CSV
    if all_results:
        df_all = pd.DataFrame(all_results)
        save_results_to_csv(df_all, "step2_optimization_results.csv")

    return best_configs


if __name__ == "__main__":
    run_step2()
