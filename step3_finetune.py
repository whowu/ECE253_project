import pandas as pd
import os
import shutil
from ultralytics import YOLO
from config import MODEL_PATH, DATASET_ROOT, TARGET_CLASSES
from utils import create_yaml, plot_metrics, apply_algorithm, split_dataset


def run_step3(best_configs):
    print("\n" + "=" * 40)
    print("STEP 3: Process vs Fine-tune")
    print("=" * 40)

    if not best_configs:
        print("No best configurations provided (Step 2 failed?). Skipping.")
        return

    results = []

    for distortion, config in best_configs.items():
        print(f"\n>>> Final Comparison for: {distortion}")

        # 1. Split Dataset (Train/Test)
        original_dist_path = os.path.join(DATASET_ROOT, f"D'/{distortion}")
        split_path = split_dataset(original_dist_path, train_ratio=0.8)

        test_imgs_path = os.path.join(split_path, "images", "test")
        test_labels_path = os.path.join(split_path, "labels", "test")

        # --- STRATEGY A: Process Test Set with Best Algo + Base Model ---
        # Apply X (Best Algo) to Test Set
        processed_test_imgs = os.path.join(split_path, "processed_test", "images")
        processed_test_labels = os.path.join(split_path, "processed_test", "labels")
        apply_algorithm(
            test_imgs_path, processed_test_imgs, config["algo"], config["param"]
        )

        # Copy labels
        if os.path.exists(processed_test_labels):
            shutil.rmtree(processed_test_labels)
        shutil.copytree(test_labels_path, processed_test_labels)

        # Evaluate Base Model on Processed Test Set
        base_model = YOLO(MODEL_PATH)
        yaml_strat_a = create_yaml(
            split_path, "processed_test/images", "strat_a"
        )  # path relative to split_path

        m_a = base_model.val(data=yaml_strat_a, verbose=False)
        for cid, cname in TARGET_CLASSES.items():
            if cid in m_a.box.ap_class_index:
                idx = list(m_a.box.ap_class_index).index(cid)
                results.append(
                    {
                        "Distortion": distortion,
                        "Strategy": "Img Processing",
                        "Class": cname,
                        "mAP@50-95": m_a.box.maps[idx],
                    }
                )
        os.remove(yaml_strat_a)

        # --- STRATEGY B: Fine-tune Model on Train Set + Eval on Unprocessed Test ---
        # Train new model
        print("   🏋️ Fine-tuning model (this may take time)...")
        ft_model = YOLO(MODEL_PATH)  # Start from pre-trained
        yaml_train = create_yaml(split_path, "images/train", "train")

        # Quick training for demo (epochs=5). Increase for real results!
        ft_model.train(data=yaml_train, epochs=5, imgsz=640, verbose=False)

        # Evaluate New Model on Unprocessed Test Set
        # We reuse the yaml but point validation to test set
        # Actually create specific test yaml
        yaml_test = create_yaml(split_path, "images/test", "test_eval")

        m_b = ft_model.val(data=yaml_test, verbose=False)
        for cid, cname in TARGET_CLASSES.items():
            if cid in m_b.box.ap_class_index:
                idx = list(m_b.box.ap_class_index).index(cid)
                results.append(
                    {
                        "Distortion": distortion,
                        "Strategy": "Fine-Tuning",
                        "Class": cname,
                        "mAP@50-95": m_b.box.maps[idx],
                    }
                )

        # Clean up yamls
        os.remove(yaml_train)
        os.remove(yaml_test)

    # Plot Comparison
    if results:
        df = pd.DataFrame(results)
        # Create separate plots for each distortion
        for distortion in best_configs.keys():
            subset = df[df["Distortion"] == distortion]
            plot_metrics(
                subset,
                f"Best Algo vs Fine-Tuning ({distortion})",
                f"step3_{distortion}.png",
                group_col="Strategy",
            )


if __name__ == "__main__":
    # Example mock input if running standalone
    mock_configs = {"noise": {"algo": "CBM3D", "param": 0.0980}}
    run_step3(mock_configs)
