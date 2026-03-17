import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================
ADE_ROOT = '/data/DatasetCollection/ADEChallengeData2016'
MASK_FOLDER_TRAIN = os.path.join(ADE_ROOT, 'annotations/training')
MASK_FOLDER_VAL = os.path.join(ADE_ROOT, 'annotations/validation')

OUTPUT_DIR = './datasets/ade/incremental_split'

# ADE20K 100-10 setting
BASE_NUM = 100
INC_NUM = 10
TOTAL_CLASSES = 150 # Raw labels 1-150 are classes

# ===========================================

def get_labels_from_mask(mask_path):
    if not os.path.exists(mask_path):
        return set()

    try:
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        unique_pixels = np.unique(mask_arr)

        labels = set()
        for p in unique_pixels:
            if p > 0 and p <= TOTAL_CLASSES:  # 0 is background/ignore
                labels.add(int(p))
        return labels

    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        return set()

def load_all_annotations(mask_folder):
    print(f"Loading masks from: {mask_folder}")
    
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]
    cache = {}
    valid_count = 0

    for f in tqdm(mask_files, desc="Reading Masks"):
        img_id = os.path.splitext(f)[0]
        mask_path = os.path.join(mask_folder, f)
        labels = get_labels_from_mask(mask_path)
        cache[img_id] = labels
        if len(labels) > 0:
            valid_count += 1

    print(f"Successfully loaded labels for {valid_count} / {len(mask_files)} images.")
    return cache

def process_setting(task_name, base_num, inc_num, all_labels_map, prefix='train'):
    print(f"\n>>> Processing Task: {task_name} ({prefix})")
    
    step_ranges = []
    # Step 1: Base classes 1 to 100
    step_ranges.append((1, base_num))
    
    # Subsequent steps: 10 classes each
    current_idx = base_num + 1
    while current_idx <= TOTAL_CLASSES:
        end_idx = min(current_idx + inc_num - 1, TOTAL_CLASSES)
        step_ranges.append((current_idx, end_idx))
        current_idx = end_idx + 1

    for step_idx, (start_idx, end_idx) in enumerate(step_ranges):
        step_num = step_idx + 1
        
        if prefix == 'train':
            # Overlap mode: images containing any class in [start_idx, end_idx]
            relevant_classes = set(range(start_idx, end_idx + 1))
            mode_str = "Overlap"
            class_str = f"{start_idx}-{end_idx}"
        else:
            # Cumulative mode: images containing any class from class 1 up to end_idx
            relevant_classes = set(range(1, end_idx + 1))
            mode_str = "Cumulative"
            class_str = f"1-{end_idx}"

        valid_images = []
        for img_id, labels in all_labels_map.items():
            if not relevant_classes.isdisjoint(labels):
                valid_images.append(img_id)

        filename = f"{prefix}_{task_name}_step_{step_num}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w') as f:
            for img_id in valid_images:
                f.write(f"{img_id}\n")

        print(f"   [{prefix.capitalize()} Step {step_num}] Classes {class_str} ({mode_str}): {len(valid_images)} images saved.")

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Training Set
    print("\n--- Processing Training Set ---")
    train_labels_map = load_all_annotations(MASK_FOLDER_TRAIN)
    process_setting('100-10', BASE_NUM, INC_NUM, train_labels_map, prefix='train')
    process_setting('offline', 150, 0, train_labels_map, prefix='train')

    # 2. Validation Set
    print("\n--- Processing Validation Set ---")
    val_labels_map = load_all_annotations(MASK_FOLDER_VAL)
    process_setting('100-10', BASE_NUM, INC_NUM, val_labels_map, prefix='val')
    process_setting('offline', 150, 0, val_labels_map, prefix='val')

    print("\nAll Done! Files saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
