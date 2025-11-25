import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F

def create_custom_dataset():
    # 1. Load the COCO Validation split
    print("Loading COCO registry...")
    dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["person", "backpack", "chair"],
    )

    # 2. Create 3 separate randomized views of 200 images each
    view_person = dataset.match(
        F("ground_truth.detections.label").contains("person")
    ).shuffle(seed=42).limit(200)

    view_backpack = dataset.match(
        F("ground_truth.detections.label").contains("backpack")
    ).shuffle(seed=42).limit(200)

    view_chair = dataset.match(
        F("ground_truth.detections.label").contains("chair")
    ).shuffle(seed=42).limit(200)

    # 3. FIX: Merge by collecting IDs instead of using '|'
    # We get the list of IDs from each view
    ids_person = view_person.values("id")
    ids_backpack = view_backpack.values("id")
    ids_chair = view_chair.values("id")

    # Combine them and use set() to remove duplicates
    # (e.g. if an image has both a person and a backpack, it appears in both lists)
    all_ids = list(set(ids_person + ids_backpack + ids_chair))

    # Create the final view using these unique IDs
    final_view = dataset.select(all_ids)

    print(f"Selected {len(ids_person)} person images")
    print(f"Selected {len(ids_backpack)} backpack images")
    print(f"Selected {len(ids_chair)} chair images")
    print(f"Total unique images in final dataset: {len(final_view)}")

    # 4. Export with explicit ID mapping
    export_dir = "datasets/D"
    
    my_classes = ["person", "backpack", "chair"]

    print(f"Exporting to {export_dir}...")
    final_view.export(
        export_dir=export_dir,
        dataset_type=fo.types.YOLOv5Dataset,
        label_field="ground_truth",
        classes=my_classes, 
    )

    print("Done.")

if __name__ == "__main__":
    create_custom_dataset()