# compute_prior_pmi_voc12_trainaug.py
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle",
    "bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor"
]  # 20前景类（不含背景）

EXPECTED_TRAIN_AUG = 10582  # 参考值，便于 sanity check

def load_trainaug_ids(voc2012_dir: Path) -> list[str]:
    seg_set_dir = voc2012_dir / "ImageSets" / "Segmentation"
    # 常见命名：trainaug.txt 或 train_aug.txt（不同repo会有差异）
    candidates = ["trainaug.txt", "train_aug.txt"]
    for fn in candidates:
        p = seg_set_dir / fn
        if p.exists():
            with p.open("r") as f:
                return [ln.strip() for ln in f if ln.strip()]
    # 兜底：直接用 SegmentationClassAug 里所有 .png 的基名
    segaug = voc2012_dir / "SegmentationClassAug"
    ids = [p.stem for p in sorted(segaug.glob("*.png"))]
    return ids

def presence_from_mask(mask_np: np.ndarray, C: int = 20) -> np.ndarray:
    """
    VOC seg mask: 0=背景, 1..20=前景类, 255=ignore
    """
    y = np.zeros((C,), dtype=np.float32)
    uniq = np.unique(mask_np)
    present = uniq[(uniq >= 1) & (uniq <= C)] - 1  # 转 0..C-1
    y[present] = 1.0
    return y

@torch.no_grad()
def compute_pmi(Y: torch.Tensor, laplace: float = 1.0, eps: float = 1e-8):
    """
    返回 (PMI, PPMI)，PMI 允许为负；PPMI = max(PMI, 0)
    归一方式与你线上实现保持一致：pair = Y^T Y；total = pair.sum()
    """
    C = Y.shape[1]
    pair = (Y.t() @ Y).float() + laplace     # [C,C] 加拉普拉斯
    total = pair.sum()                       # 标量
    Pij = pair / total                       # [C,C]
    Pi  = torch.diag(pair) / total           # [C]

    denom = (Pi.view(C,1) * Pi.view(1,C)).clamp_min(eps)
    pmi   = torch.log((Pij + eps) / denom)   # 允许负值
    pmi.fill_diagonal_(0.0)
    pmi  = 0.5 * (pmi + pmi.t())             # 对称化

    ppmi = torch.clamp(pmi, min=0.0)         # 兼容用
    return pmi.cpu(), ppmi.cpu()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voc2012_dir", type=str, default="/data/DatasetCollection/VOCdevkit/VOC2012",
                    help="例如 /data/DatasetCollection/VOCdevkit/VOC2012")
    ap.add_argument("--laplace", type=float, default=1.0,
                    help="拉普拉斯平滑系数(0.5~2.0)")
    ap.add_argument("--out", type=str, default="prior_pmi_voc12_trainaug.pt",
                    help="输出文件")
    ap.add_argument("--strict10582", action="store_true",
                    help="若数量不是 10582 则报错")
    args = ap.parse_args()

    voc2012 = Path(args.voc2012_dir)
    segaug = voc2012 / "SegmentationClassAug"
    assert voc2012.exists(), f"{voc2012} 不存在"
    assert segaug.exists(), f"{segaug} 不存在"

    ids = load_trainaug_ids(voc2012)
    # 只保留在 SegmentationClassAug 里确实有掩码的 id
    ids = [i for i in ids if (segaug / f"{i}.png").exists()]

    n = len(ids)
    msg = f"Found {n} trainaug ids with masks in SegmentationClassAug."
    if args.strict10582 and n != EXPECTED_TRAIN_AUG:
        raise RuntimeError(msg + f" (expected {EXPECTED_TRAIN_AUG})")
    else:
        print(msg)

    Ys = []
    for img_id in ids:
        m = np.array(Image.open(segaug / f"{img_id}.png"), dtype=np.uint8)
        Ys.append(presence_from_mask(m, C=len(VOC_CLASSES)))
    Y = torch.from_numpy(np.stack(Ys, axis=0))  # [N,20]

    pmi, ppmi = compute_pmi(Y, laplace=args.laplace)  # ★ 现在主用 PMI

    torch.save({
        "prior_pmi":  pmi,                # ★ 主输出：PMI（允许负）
        "prior_ppmi": ppmi,               # 兼容/可视化需要时也在
        "classes": VOC_CLASSES,
        "used_ids": ids,
        "split": "trainaug",
        "laplace": args.laplace,
        "source": "VOC2012/SegmentationClassAug (trainaug)",
        "count": n,
    }, args.out)

    print(f"Saved to {args.out}. PMI/PPMI shape: {tuple(pmi.shape)}")

if __name__ == "__main__":
    main()
