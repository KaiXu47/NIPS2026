import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# ================= 配置区域 =================

# 【修改 1】: 将 VOC_ROOT 改为存放数据的绝对路径
# 根据你的 SOURCE_TXT_PATH 推测，你的数据应该在这里：
VOC_ROOT = '/data/DatasetCollection/VOCdevkit/VOC2012'

# 2. 掩码文件夹名称
MASK_FOLDER_NAME = 'SegmentationClassAug'

# 3. 源列表文件
TRAIN_SOURCE_TXT_PATH = '/data/DatasetCollection/VOCdevkit/VOC2012/list/train_aug.txt'
VAL_SOURCE_TXT_PATH = '/data/DatasetCollection/VOCdevkit/VOC2012/list/val.txt'

# 4. 输出文件夹
OUTPUT_DIR = './datasets/voc/incremental_split'

# 5. 配置列表
SETTINGS = [
    (15, 5),
    (10, 10),
    (10, 1),
    (15, 1),
    (5, 3),
    (2, 2),
    (1, 1)
]

# 6. VOC 类别顺序
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


# ===========================================

def get_labels_from_mask(voc_root, image_id):
    """
    读取 PNG 掩码文件
    """
    # 尝试构建路径
    mask_path = os.path.join(voc_root, MASK_FOLDER_NAME, f'{image_id}.png')

    if not os.path.exists(mask_path):
        # 回退尝试：有时候 SegmentationClassAug 只有部分，剩下的在 SegmentationClass
        fallback_path = os.path.join(voc_root, 'SegmentationClass', f'{image_id}.png')
        if os.path.exists(fallback_path):
            mask_path = fallback_path
        else:
            # 【调试信息】：如果找不到，打印出来看看路径到底拼对没有
            # 只打印前3个错误，避免刷屏
            if get_labels_from_mask.error_count < 3:
                print(f"\n[Error] Mask not found: {mask_path}")
                print(f"       -> Please check VOC_ROOT or MASK_FOLDER_NAME.")
                get_labels_from_mask.error_count += 1
            return set()

    try:
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        unique_pixels = np.unique(mask_arr)

        labels = set()
        for p in unique_pixels:
            if p > 0 and p != 255:  # 排除背景(0)和边界(255)，保留 1-20 作为类别标签
                labels.add(int(p))
        return labels

    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        return set()


# 初始化错误计数器
get_labels_from_mask.error_count = 0


def clean_image_id(line):
    """
    【修改 2】: 健壮的 ID 解析函数
    处理 train_aug.txt 可能出现的各种奇葩格式
    例如: "/JPEGImages/2007_000032.jpg /SegmentationClass/2007_000032.png"
    或者: "2007_000032"
    """
    line = line.strip()
    # 1. 如果有空格（多列），只取第一列
    parts = line.split()
    item = parts[0]

    # 2. 去掉路径和后缀，只保留文件名（不含扩展名）
    # 例如: "/path/to/2007_000032.jpg" -> "2007_000032"
    filename = os.path.basename(item)
    image_id = os.path.splitext(filename)[0]

    return image_id


def load_all_annotations(voc_root, image_ids):
    print(f"Loading masks for {len(image_ids)} images...")
    print(f"Looking for masks in: {os.path.join(voc_root, MASK_FOLDER_NAME)}")

    cache = {}
    valid_count = 0

    for img_id in tqdm(image_ids, desc="Reading Masks"):
        labels = get_labels_from_mask(voc_root, img_id)
        cache[img_id] = labels
        if len(labels) > 0:
            valid_count += 1

    print(f"Successfully loaded labels for {valid_count} / {len(image_ids)} images.")
    if valid_count == 0:
        print("!!! CRITICAL WARNING: No labels were loaded. Check paths above !!!")

    return cache


def process_setting(base_num, inc_num, all_labels_map, prefix='train'):
    print(f"\n>>> Processing Setting: Base={base_num}, Increment={inc_num} ({prefix})")
    total_classes = 20
    step_ranges = []
    # Base step: classes 1 to base_num
    step_ranges.append((1, base_num))
    
    current_idx = base_num + 1
    while current_idx <= total_classes:
        end_idx = min(current_idx + inc_num - 1, total_classes)
        step_ranges.append((current_idx, end_idx))
        current_idx = end_idx + 1

    for step_idx, (start_idx, end_idx) in enumerate(step_ranges):
        step_num = step_idx + 1
        
        if prefix == 'train':
            # 训练集 (Overlap 模式)：仅包含含有当前增量步类别的图片
            relevant_classes = set(range(start_idx, end_idx + 1))
            mode_str = "Overlap"
            class_str = f"{start_idx}-{end_idx}"
        else:
            # 验证集 (叠加/累积模式)：包含目前为止所有见过的前景类别 (从 1 开始)
            relevant_classes = set(range(1, end_idx + 1))
            mode_str = "Cumulative"
            class_str = f"1-{end_idx}"

        valid_images = []
        for img_id, labels in all_labels_map.items():
            if not relevant_classes.isdisjoint(labels):
                valid_images.append(img_id)

        filename = f"{prefix}_{base_num}-{inc_num}_step_{step_num}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, 'w') as f:
            for img_id in valid_images:
                f.write(f"{img_id}\n")

        print(f"   [{prefix.capitalize()} Step {step_num}] Classes {class_str} ({mode_str}): {len(valid_images)} images saved.")


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 处理训练集
    print("\n--- Processing Training Set ---")
    if os.path.exists(TRAIN_SOURCE_TXT_PATH):
        with open(TRAIN_SOURCE_TXT_PATH, 'r') as f:
            train_ids = [clean_image_id(x) for x in f.readlines() if x.strip()]
        train_labels_map = load_all_annotations(VOC_ROOT, train_ids)
        for base, inc in SETTINGS:
            process_setting(base, inc, train_labels_map, prefix='train')
    else:
        print(f"Warning: {TRAIN_SOURCE_TXT_PATH} not found.")

    # 2. 处理验证集
    print("\n--- Processing Validation Set ---")
    if os.path.exists(VAL_SOURCE_TXT_PATH):
        with open(VAL_SOURCE_TXT_PATH, 'r') as f:
            val_ids = [clean_image_id(x) for x in f.readlines() if x.strip()]
        val_labels_map = load_all_annotations(VOC_ROOT, val_ids)
        for base, inc in SETTINGS:
            process_setting(base, inc, val_labels_map, prefix='val')
    else:
        print(f"Warning: {VAL_SOURCE_TXT_PATH} not found.")

    print("\nAll Done! Files saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()