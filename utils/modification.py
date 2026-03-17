import pdb
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt



def remove_cow(cls_label, labels):
    cls_label[:, -1] = 0
    labels[labels == 10] = 255
    return cls_label, labels


import numpy as np
import matplotlib.pyplot as plt

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
    'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor', 'ignore'
]


# VOC 21 类 + ignore (255) 对应 RGB 颜色
VOC_COLORS = np.array([
    [0, 0, 0],        # 0 background
    [128, 0, 0],      # 1 aeroplane
    [0, 128, 0],      # 2 bicycle
    [128, 128, 0],    # 3 bird
    [0, 0, 128],      # 4 boat
    [128, 0, 128],    # 5 bottle
    [0, 128, 128],    # 6 bus
    [128, 128, 128],  # 7 car
    [64, 0, 0],       # 8 cat
    [192, 0, 0],      # 9 chair
    [64, 128, 0],     # 10 cow
    [192, 128, 0],    # 11 diningtable
    [64, 0, 128],     # 12 dog
    [192, 0, 128],    # 13 horse
    [64, 128, 128],   # 14 motorbike
    [192, 128, 128],  # 15 person
    [0, 64, 0],       # 16 potted plant
    [128, 64, 0],     # 17 sheep
    [0, 192, 0],      # 18 sof  a
    [128, 192, 0],    # 19 train
    [0, 64, 128],     # 20 tv/monitor
    [192, 192, 192]   # 255 ignore → 灰色
], dtype=np.uint8)

fig, ax = plt.subplots(figsize=(4, 8))
for i, (cls, color) in enumerate(zip(VOC_CLASSES, VOC_COLORS)):
    ax.add_patch(plt.Rectangle((0, i), 1, 1, color=color/255.0))
    ax.text(1.05, i + 0.5, cls, va='center', fontsize=10)

ax.set_xlim(0, 3)
ax.set_ylim(0, len(VOC_CLASSES))
ax.axis('off')
plt.tight_layout()
plt.savefig("voc_label_colors.png")
plt.close()


def label_to_color(label_map):
    """
    label_map: [H, W]，类别 0~20 或 255
    return: [H, W, 3] RGB
    """
    color_map = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
    mask_ignore = (label_map == 255)

    # 0~20 标签映射
    for cid in range(21):
        color_map[label_map == cid] = VOC_COLORS[cid]

    # ignore 标签
    color_map[mask_ignore] = VOC_COLORS[21]

    return color_map
# ---------------------------------------------------------
def visualize_dense_labels(cam, segs, mixed_label, idx=0):
    """
    cam, segs, mixed_label 均为 [B,H,W] 的 dense label
    idx 表示 batch 中第几个
    """
    cam_color = label_to_color(cam[idx].cpu().numpy().astype(np.int32))
    seg_color = label_to_color(segs[idx].cpu().numpy().astype(np.int32))
    mix_color = label_to_color(mixed_label[idx].cpu().numpy().astype(np.int32))

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("CAM label")  
    plt.imshow(cam_color)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Segs")
    plt.imshow(seg_color)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Mixed Label")
    plt.imshow(mix_color)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{idx}.png")
    plt.close()

def filter_dense_labels(old_pixel_label, cls_label, ignore_index=255):
    """
    old_pixel_label: [B,H,W] long tensor
    cls_label: [B,C] binary or multi-hot, 表示每张图哪些类别存在 (0/1)
    ignore_index: 要清除的像素赋值
    
    返回: [B,H,W] 清洗后的 dense label
    """
    B, H, W = old_pixel_label.shape
    new_label = old_pixel_label.clone()

    for b in range(B):

        present_classes = torch.nonzero(cls_label[b], as_tuple=False).squeeze(1)
        present_classes = present_classes + 1
        present_classes = torch.cat([torch.tensor([0]).cuda(), present_classes])

        mask = ~torch.isin(new_label[b], present_classes)
        
        new_label[b][mask] = ignore_index

    return new_label


def get_class_weight(cls_label, new_classes):
    num_classes = cls_label.size(1)

    class_weight = torch.full((num_classes,), 1.0, device=cls_label.device)

    class_weight[:-new_classes] = 0.2

    return class_weight

def masked_avg_pooling(fmap, gt_mask, total_classes , H=112, W=112):
    """
    fmap: (B, D, H, W)
    gt_mask: (B, H, W)
    return: (total_classes, D), 没出现的类返回全零
    """
    B, D, _, _= fmap.shape
    # _, H, W = gt_mask.shape


    gt_mask = F.interpolate(gt_mask.unsqueeze(1), size=(H, W), mode="nearest").squeeze(1)
    fmap = F.interpolate(fmap, size=(H, W), mode="bilinear", align_corners=True)

    # print(gt_mask.shape)
    fmap = fmap.reshape(B, D, -1)        # (B, D, HW)
    gt_mask = gt_mask.reshape(B, -1)     # (B, HW)
    
    out = torch.zeros(total_classes, D, device=fmap.device)
    
    for c in gt_mask.unique():
        if (c == 255) | (c == 0):
            continue
        c = int(c)
        mask = (gt_mask == c).float()        # (B, HW)
        denom = mask.sum() + 1e-6
        mask = mask.unsqueeze(1)             # (B, 1, HW)
        
        feat = (fmap * mask).sum(dim=[0, 2]) / denom
        out[c-1] = feat

    return out


def prototype_separation_loss(p_iter, P_pool, margin=0.3):
    """
    p_iter: (total_classes, D) masked avg pooled prototype
    P_pool: (total_classes, D) current prototype pool
    """
    loss = torch.tensor(0.0).cuda()
    total_classes = p_iter.size(0)

    P_pool_norm = F.normalize(P_pool, dim=1)
    p_iter_norm = F.normalize(p_iter, dim=1)
    cnt = 0

    for c in range(total_classes):
        
        if p_iter[c].sum() == 0:
            continue

        pos = 1 - F.cosine_similarity(p_iter_norm[c], P_pool_norm[c], dim=0)

        # 与其他类远离：cosine <= margin
        negs = []
        for k in range(total_classes):
            if k == c:
                continue
            cos = F.cosine_similarity(p_iter_norm[c], P_pool_norm[k], dim=0)
            negs.append(F.relu(cos - margin))   # pushing away

        neg = torch.stack(negs).mean()

        loss += pos + neg
        cnt += 1

    return loss / (cnt+1e-6)


def get_type_seg_loss(logits, target, ignore_index=255, keep_ratio=1.0):
    """
    带 OHEM (Online Hard Example Mining) 的交叉熵损失。
    keep_ratio=1.0 时退化为标准 Cross Entropy。
    """
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(
            logits, 
            size=target.shape[-2:], 
            mode='bilinear', 
            align_corners=False
        )
    
    # 1. 计算每个像素的 Cross Entropy (reduction='none' 保持像素独立)
    ce_loss = F.cross_entropy(logits, target.long(), ignore_index=ignore_index, reduction='none')
    
    # 2. 扁平化以便排序
    ce_loss = ce_loss.view(-1)
    
    # 3. 过滤掉 ignore 区域 (ce_loss 默认为 0，但为了严谨我们显式过滤)
    valid_mask = ce_loss > 0
    ce_loss = ce_loss[valid_mask]
    
    if ce_loss.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # 4. 执行 OHEM 排序与筛选
    # 计算需要保留的难样本数量
    num_keep = int(keep_ratio * ce_loss.numel())
    if num_keep < ce_loss.numel():
        # 找到前 num_keep 个最大的 loss 值
        vals, _ = torch.topk(ce_loss, num_keep)
        return vals.mean()
    
    return ce_loss.mean()

def get_dice_loss(logits, target, ignore_index=255, smooth=1.0):
    """
    多类别 Dice Loss，对类别不均衡更鲁棒。
    """
    if logits.shape[-2:] != target.shape[-2:]:
        logits = F.interpolate(logits, size=target.shape[-2:], mode='bilinear', align_corners=False)
    
    B, C, H, W = logits.shape
    probs = F.softmax(logits, dim=1)
    
    # 1. 有效区域 Mask
    valid_mask = (target != ignore_index)
    
    # 2. 安全构建 One-Hot
    # 防止 target 中的 ignore_index (如 255) 导致 one_hot 越界崩溃
    target_safe = torch.clamp(target, 0, C - 1).long()
    target_one_hot = F.one_hot(target_safe, num_classes=C).permute(0, 3, 1, 2).float()
    
    # 3. 过滤掉 ignore_index 区域的影响
    valid_mask_float = valid_mask.unsqueeze(1).float()
    probs = probs * valid_mask_float
    target_one_hot = target_one_hot * valid_mask_float
    
    # 4. 计算 Intersection 和 Cardinality (在 B, H, W 维度上求和)
    dims = (0, 2, 3) 
    intersection = torch.sum(probs * target_one_hot, dim=dims)
    cardinality = torch.sum(probs + target_one_hot, dim=dims)
    
    # 5. Dice Score & Loss
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth)
    dice_loss = 1.0 - dice_score
    
    # 6. 动态平均：只计算当前 batch 中真实存在的类别
    # 避免图片中根本没有的类别产生 1.0 的满额 loss 干扰训练
    present_mask = (target_one_hot.sum(dim=dims) > 0)
    
    if present_mask.sum() > 0:
        return dice_loss[present_mask].mean()
        
    return dice_loss.mean()

def prototype_kd_loss(p_new, p_old, lambda_dir=1.0, lambda_struct=1.0, skip_bg=True):
    """
    p_old: [C, D] old model prototypes（teacher）
    p_new: [C, D] new model prototypes（student）
    """
    C = p_old.size(0)
    old_C = C
    
    if skip_bg:
        p_old = p_old[1:old_C]
        p_new = p_new[1:old_C]
    else:
        p_old = p_old[:old_C]
        p_new = p_new[:old_C]

    # print(p_old.shape)
    # print(p_new.shape)
    
    # ---- 2. L2 normalization ----
    p_old_norm = F.normalize(p_old, dim=1)
    p_new_norm = F.normalize(p_new, dim=1)

    
    
    # ============================================================
    # (A) Direction KD: preserve each prototype direction
    # ============================================================
    cos_diag = F.cosine_similarity(p_old_norm, p_new_norm)
    # print(cos_diag)
    loss_dir = (2 * (1 - cos_diag)).mean()
    # print(loss_dir)
    # ============================================================
    # (B) Structure KD: preserve inter-class cosine matrix
    # ============================================================
    old_cos = torch.matmul(p_old_norm, p_old_norm.t())  # [C_old, C_old]
    new_cos = torch.matmul(p_new_norm, p_new_norm.t())
    # print(old_cos,new_cos)
    loss_struct = F.mse_loss(new_cos, old_cos)
    # print(loss_struct)
    # ============================================================
    # Final loss
    # ============================================================
    loss = lambda_dir * loss_dir + lambda_struct * loss_struct

    return loss
 
def prototype_sep_loss(p_new_raw, cls_label , num_new_cls = 0, skip_bg=True):
    
    if skip_bg:
        p_new = p_new_raw[1:]
    else:
        p_new = p_new_raw
    device = p_new.device
    C, D = p_new.size()

    old_idx = torch.arange(0, C - num_new_cls).to(device)
    new_idx = torch.arange(C - num_new_cls, C).to(device)

    # ---------------------------
    # Case A: 没有新类 → 全部旧类 prototype 分开
    # ---------------------------
    if num_new_cls == 0:
        # cosine similarity matrix CxC
        sim = F.cosine_similarity(p_new.unsqueeze(1), p_new.unsqueeze(0), dim=-1)
        # 只取非对角元素
        mask = ~torch.eye(C, dtype=torch.bool, device=device)
        return sim[mask].mean()

    # ---------------------------
    # Case B: 存在新类
    # ---------------------------
    final_loss = 0.0
    count = 0

    for b in range(cls_label.size(0)):
        present = cls_label[b].bool()

        present_new = new_idx[present[new_idx]]
        present_old = old_idx[present[old_idx]]

        # 情况 1：仅包含新类 → 新类与所有旧类分离（weight α）
        if len(present_old) == 0:
            if len(present_new) == 0:
                continue
            n_proto = p_new[present_new]           # [N_new, D]
            o_proto = p_new[old_idx]               # [N_old_all, D]
            sim = F.cosine_similarity(
                n_proto.unsqueeze(1), o_proto.unsqueeze(0), dim=-1
            )
            final_loss += 0.5 * F.relu(sim).mean()
            count += 1

        # 情况 2：包含新类 + 旧类 → 新类和“对应旧类”分离
        else:
            n_proto = p_new[present_new]           # [N_new, D]
            o_proto = p_new[present_old]           # [N_old_present, D]
            sim = F.cosine_similarity(
                n_proto.unsqueeze(1), o_proto.unsqueeze(0), dim=-1
            )
            final_loss += F.relu(sim).mean()
            count += 1
        
        # n_proto = p_new[present_new]             
        # bg_proto = p_new_raw[0]
        # sim = F.cosine_similarity(
        #     n_proto.unsqueeze(1), bg_proto.unsqueeze(0), dim=-1
        # )
        # final_loss += F.relu(sim).mean()

    if count == 0:
        return torch.tensor(0.0, device=device)

    return final_loss / count


def margin_triplet_peaky_loss(type_seg_logits, margin=0.3):
    """
    type_seg_logits: [B, C, H, W], values in [0, 1]
    margin: required top1 >= top2 + margin
    """

    B, C, H, W = type_seg_logits.shape
    

    
    # flatten to [N, C]
    logits = type_seg_logits.permute(0, 2, 3, 1).reshape(-1, C)   # [B*H*W, C]
    
    type_seg_rst = logits.argmax(dim=1)
    mask_fg = (type_seg_rst != 0)
    
    # top1
    top1_vals, top1_idx = logits.max(dim=1)    # [N]

    # mask out top1 to find top2
    logits_clone = logits.clone()
    logits_clone[torch.arange(logits.size(0)), top1_idx] = -1e9
    top2_vals, _ = logits_clone.max(dim=1)

    # triplet margin: enforce top1 >= top2 + margin
    loss = (margin + top2_vals[mask_fg] - top1_vals[mask_fg]).clamp(min=0)

    return loss.mean()


def compute_dynamic_margin_ortho_loss(p_current, clip_sim_matrix, max_margin=0.6):
    """
    带上限的动态 Margin 正交损失：
    p_current: [C, D] 当前模型的 Prototype
    clip_sim_matrix: [C, C] 离线的 CLIP 相似度矩阵
    max_margin: 容忍度的上限。即使 CLIP 相似度是 0.8，Margin 也只给到 0.4，确保必须有 0.6 的区分度。
    """
    C = p_current.size(0)
    p_n = F.normalize(p_current, dim=-1)
    # 1. 计算当前 Prototype 间的相似度
    sim_matrix = torch.mm(p_n, p_n.t()) # [C, C]
    
    # 2. 计算动态 Margin：取 CLIP 相似度和上限的最小值
    # clip_sim_matrix 应预先对齐到当前类别数
    dynamic_margin = torch.clamp(clip_sim_matrix, max=max_margin)
    
    # 3. 只有当 当前相似度 > 动态允许的相似度(Margin) 时，才产生惩罚
    loss = torch.pow(F.relu(sim_matrix - dynamic_margin), 2)
    
    # 排除对角线（类自身）
    mask = 1.0 - torch.eye(C, device=p_current.device)
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def compute_comprehensive_ortho_loss(
    p_global, 
    p_final, 
    delta_p, 
    mode='all',
    margin=0.2 
):
    """
    Soft-margin Orthogonality Loss:
    允许类别之间存在一定的余弦相似度 (小于 margin 时 loss 为 0)。
    这样可以保留类别之间的语义共性，只压制过高的冗余。
    """
    B, C, D = p_final.shape
    loss_ortho = 0.0

    # 1. 基础归一化
    p_global_n = F.normalize(p_global, dim=-1)     # [C, D]
    p_final_n = F.normalize(p_final, dim=-1)       # [B, C, D]

    # 为了避免对角线（类自身）的干扰，创建一个 mask
    mask = (1.0 - torch.eye(C, device=p_global.device)) # [C, C]

    if mode == 'global_only' or mode == 'all':
        # 全局原型之间的相似度矩阵
        sim_g = torch.mm(p_global_n, p_global_n.t()) # [C, C]
        
        # 软间隔控制：只有当相似度 > margin 时才产生惩罚
        # 使用 pow(2) 增加大偏差的惩罚力度
        loss_g = torch.pow(F.relu(sim_g * mask - margin), 2).mean()
        loss_ortho += loss_g

    if mode == 'cross' or mode == 'all':
        # p_final_n: [B, C, D], p_global_n: [C, D]
        # 计算当前 instance 原型与全局原型的交叉相似度
        sim_cross = torch.matmul(p_final_n, p_global_n.t()) # [B, C, C]
        
        mask_cross = mask.unsqueeze(0) # [1, C, C]
        loss_c = torch.pow(F.relu(sim_cross * mask_cross - margin), 2).mean()
        loss_ortho += loss_c

    # 残差项的范数正则（如果你现在用了 SimpleHead，delta_p 是 0，这项会自动失效）
    loss_reg = torch.mean(torch.norm(delta_p, p=2, dim=-1))

    return loss_ortho + 0.1 * loss_reg