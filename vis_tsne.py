import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import seaborn as sns
import pandas as pd
import os

# 定义 VOC 类别名称
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def balance_classes(X, y, samples_per_class=400):
    print(f"Balancing classes: aiming for {samples_per_class} points per class...")
    new_X = []
    new_y = []

    unique_classes = np.unique(y)
    for cls in unique_classes:
        indices = np.where(y == cls)[0]
        count = len(indices)
        if count == 0: continue

        if count > samples_per_class:
            np.random.seed(42)
            selected_indices = np.random.choice(indices, samples_per_class, replace=False)
        else:
            selected_indices = indices

        new_X.append(X[selected_indices])
        new_y.append(y[selected_indices])

    if len(new_X) == 0:
        return np.array([]), np.array([])

    return np.concatenate(new_X, axis=0), np.concatenate(new_y, axis=0)

def visualize_tsne(step_id, feature_path, label_path):
    print(f"Loading data from step {step_id}...")
    try:
        X = np.load(feature_path)
        y = np.load(label_path)
    except FileNotFoundError:
        print(f"Error: File not found. Check path: {feature_path}")
        return

    # 1. 清洗无效数据
    if np.isnan(X).any() or np.isinf(X).any():
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]
        y = y[mask]

    # ==========================================================
    # [设置] 保留所有前景类别 (1-20)
    # ==========================================================
    mask_valid = (y > 0) & (y < len(VOC_CLASSES))
    X = X[mask_valid]
    y = y[mask_valid]

    print(f"Classes found: {np.unique(y)}")

    if len(X) == 0:
        print("Error: No data found.")
        return

    # 2. 平衡采样
    X, y = balance_classes(X, y, samples_per_class=300)

    # ==========================================================
    # [关键修改] Ours/SASA 设置: 归一化
    # ==========================================================

    # ==========================================================
    # [t-SNE] Ours 参数 (增强分离版)
    # ==========================================================
    print("Running t-SNE (SASA Config - Enhanced Separation)...")
    tsne = TSNE(
        n_components=2,
        init='pca',
        learning_rate='auto',
        random_state=42,
        n_jobs=-1,
        metric='euclidean',
        # [修改 1] 增加夸张系数：默认是12，改为24可以让簇与簇之间分得更开，空白更多
        early_exaggeration=12,
        # [修改 2] 标准困惑度：30通常比45能产生更紧凑的局部簇
        perplexity=20,
        n_iter=1500
    )
    X_embedded = tsne.fit_transform(X)

    # ==========================================================
    # [绘图准备]
    # ==========================================================
    class_names = [VOC_CLASSES[int(i)] for i in y]

    df = pd.DataFrame({
        'dim1': X_embedded[:, 0],
        'dim2': X_embedded[:, 1],
        'label': y,
        'Class': class_names
    })

    # ==========================================================
    # 绘图
    # ==========================================================
    # 调整画布大小，稍微宽一点适应两列图例
    plt.figure(figsize=(15, 10))

    unique_labels = sorted(np.unique(y))
    n_classes = len(unique_labels)

    base_palette = sns.color_palette("husl", n_classes)

    name_palette = {}
    for i, lbl_id in enumerate(unique_labels):
        name = VOC_CLASSES[int(lbl_id)]
        name_palette[name] = base_palette[i]

    sns.scatterplot(
        data=df,
        x='dim1', y='dim2',
        hue='Class',
        hue_order=[VOC_CLASSES[int(i)] for i in unique_labels],
        palette=name_palette,
        legend='full',
        s=30,       # 点稍微大一点
        alpha=0.9,  # [修改] 透明度更高，显得颜色更实，减少视觉上的混杂感
        linewidth=0
    )

    plt.title(f"Class Feature Distribution (All Classes) - Baseline", fontsize=24, weight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=18)
    plt.ylabel("t-SNE Dimension 2", fontsize=18)
    plt.xticks([])
    plt.yticks([])

    # ==========================================================
    # [图例] 修改为两列 (ncol=2)
    # ==========================================================
    plt.legend(
        bbox_to_anchor=(1.01, 1),
        loc='upper left',
        borderaxespad=0,
        title="Classes",
        ncol=2,             # [修改] 改为两列
        fontsize=15,        # 稍微调小字体以适应两列
        title_fontsize=18,
        markerscale=2.0,
        labelspacing=0.8,
        columnspacing=1.0,  # 列间距
        handletextpad=0.5,
        frameon=False
    )

    plt.tight_layout()

    save_path = f"tsne_step_{step_id}_sasa_all_classes_sep_enhanced.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    STEP = 5
    FEAT_FILE = f"analysis_results/features_step_{STEP}.npy"
    LABEL_FILE = f"analysis_results/labels_step_{STEP}.npy"

    if os.path.exists(FEAT_FILE):
        visualize_tsne(STEP, FEAT_FILE, LABEL_FILE)
    else:
        print(f"File NOT found: {FEAT_FILE}")