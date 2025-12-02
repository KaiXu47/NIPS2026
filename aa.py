import torch
import numpy as np

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]

ckpt = torch.load("prior_pmi_voc12_trainaug.pt", map_location="cpu")
M = ckpt["prior_pmi"].cpu().numpy()              # [20,20]
classes = ckpt.get("classes", VOC_CLASSES)

# 打印为对齐的表格
w = 8  # 每个单元格宽度
print(" " * w + "".join(f"{c[:6]:>{w}}" for c in classes))
for i, ci in enumerate(classes):
    row = "".join(f"{M[i, j]:{w}.3f}" for j in range(len(classes)))
    print(f"{ci[:6]:>{w}}{row}")

# （可选）另存 CSV，便于查看
try:
    import pandas as pd
    pd.DataFrame(M, index=classes, columns=classes).to_csv("prior_ppmi_voc12_trainaug.csv", float_format="%.6f")
    print("\nSaved CSV: prior_ppmi_voc12_trainaug.csv")
except Exception:
    pass