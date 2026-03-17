import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as encoder
import numpy as np



class IncrementalClassifier(nn.ModuleList):
    def forward(self, input):
        out = []
        for mod in self:
            out.append(mod(input))
        sem_logits = torch.cat(out, dim=1)
        return sem_logits

    def get_weight(self):
        return torch.cat([mod.weight for mod in self], dim=0)

class IncrementalPrototype(nn.Module):
    """
    Incrementally adds prototype tensors for new classes.
    """
    def __init__(self, prototypes_per_task, feat_dim, init='sphere'):
        super().__init__()
        self.init = init
        self.feat_dim = feat_dim

        self.prototype = nn.ParameterList()
        for c in prototypes_per_task:
            self.prototype.append(nn.Parameter(self._init_prototype(c)))

    def _init_prototype(self, num_classes):
        if self.init == 'sphere':
            # 随机初始化并投影到单位球面上，适合 Cosine Similarity
            w = torch.randn(num_classes, self.feat_dim)
            return F.normalize(w, p=2, dim=-1)
        elif self.init == 'normal':
            return torch.randn(num_classes, self.feat_dim) * 0.01
        elif self.init == 'xavier':
            w = torch.empty(num_classes, self.feat_dim)
            nn.init.xavier_normal_(w)
            return w
        elif self.init == 'zeros':
            return torch.zeros(num_classes, self.feat_dim)
        else:
            raise ValueError(f"Unknown init: {self.init}")

    def forward(self):
        return torch.cat(list(self.prototype), dim=0)


class ResidualPrototypeModel(nn.Module):
    def __init__(self, dim): # 注意：这里不需要传 num_classes 了
        super().__init__()
        
        # 1. 全局静态原型 (The Anchor)
        # 注意：这个 Parameter 还是需要在外部或者这里动态维护，但 Adapter 本身不再依赖它
        # 如果你希望在这里管理 P，可以写一个方法来动态扩展它，见后文
        
        # 2. 残差生成器 (The Adapter)
        # 改动点：现在的 Adapter 接收 (Image_Feat + Prototype) 作为输入
        # 输入维度: dim (图像) + dim (原型) = 2 * dim
        # 输出维度: dim (即 delta_p)
        # 这样参数量就是固定的了！
        self.res_adapter = nn.Sequential(
            nn.Linear(dim * 2, dim),     # 融合 图像 和 原型
            nn.LayerNorm(dim),           # 加个 Norm 训练更稳
            nn.ReLU(),
            nn.Linear(dim, dim // 2),    # 瓶颈层
            nn.ReLU(),
            nn.Linear(dim // 2, dim)     # 输出 delta
        )
        # 这种初始化保证初始 delta 为 0 (可选，有助于初期稳定)
        nn.init.zeros_(self.res_adapter[-1].weight)
        nn.init.zeros_(self.res_adapter[-1].bias)

        self.dim = dim

    def forward(self, fmap, global_prototypes):
        """
        fmap: [B, D, H, W]
        global_prototypes: [C, D]  <-- C 可以是任意数量
        """
        B, D, H, W = fmap.shape
        C = global_prototypes.shape[0] # 动态获取当前的类别数

        # 1. 提取图像的全局上下文 [B, D]
        img_context = F.adaptive_avg_pool2d(fmap, (1, 1)).flatten(1)
        
        # 2. 准备拼接输入
        # 我们需要让 Batch 里的每张图，都和所有的 Prototype 进行交互
        
        # img_context: [B, D] -> [B, C, D] (复制 C 份)
        img_expanded = img_context.unsqueeze(1).expand(-1, C, -1)
        
        # prototypes: [C, D] -> [B, C, D] (复制 B 份)
        proto_expanded = global_prototypes.unsqueeze(0).expand(B, -1, -1)
        
        # 拼接: [B, C, 2*D]
        combined_feat = torch.cat([img_expanded, proto_expanded], dim=-1)
        
        # 3. 通过 Class-Agnostic Adapter 生成残差
        # Input: [B, C, 2*D] -> MLP -> Output: [B, C, D]
        delta_p = self.res_adapter(combined_feat)
        
        # --- 叠加 ---
        # Final P = Anchor + Delta
        p_final = proto_expanded + delta_p
        
        # --- 分割预测 ---
        p_final_norm = F.normalize(p_final, dim=2)
        fmap_norm = F.normalize(fmap.flatten(2), dim=1)
        logits = torch.bmm(p_final_norm, fmap_norm).view(B, C, H, W)
        
        return logits, delta_p, p_final


class SimpleCosineSimHead(nn.Module):
    """
    Simplified Cosine Similarity Head:
    1. L2 Normalizes both features and prototypes.
    2. Performs dot product for classification.
    3. Uses a learnable temperature (logit_scale).
    """
    def __init__(self, dim):
        super().__init__()
        # Learnable logit scale, initialized to 1/0.1 = 10 (scale = exp(logit_scale))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.1))

    def forward(self, fmap, global_prototypes):
        B, D, H, W = fmap.shape
        C = global_prototypes.shape[0]

        # 1. L2 Normalize Image Features and Prototypes
        fmap_norm = F.normalize(fmap, p=2, dim=1) # [B, D, H, W]
        proto_norm = F.normalize(global_prototypes, p=2, dim=1) # [C, D]

        # 2. Compute Cosine Similarity
        # fmap_norm: [B, D, H*W]
        fmap_flat = fmap_norm.view(B, D, H * W)
        # dot product: [B, C, H*W]
        logits = torch.matmul(proto_norm, fmap_flat).view(B, C, H, W)
        
        # 3. Apply Temperature Scale
        scale = self.logit_scale.exp()
        logits = logits * scale

        # For compatibility with the current Trainer return signature:
        # We return a zero delta_p and p_final as just expanded global_prototypes
        delta_p = torch.zeros(B, C, D, device=fmap.device)
        p_final = global_prototypes.unsqueeze(0).expand(B, -1, -1)
        
        return logits, delta_p, p_final

# ------------------------------
# Network with simplified Cosine Head
# ------------------------------

class network(nn.Module):
    def __init__(self, backbone, num_classes=None, classes_list=None, pretrained=None, init_momentum=None,
                 aux_layer=None, step=0):
        super().__init__()
        self.num_classes = num_classes
        self.classes_list = classes_list
        self.init_momentum = init_momentum

        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer, step=step)

        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [
            self.encoder.embed_dims[-1]
        ] * 4

        self.pooling = F.adaptive_max_pool2d

        # P 的参数仍由 IncrementalPrototype 管理
        self.prototype_module = IncrementalPrototype(
            prototypes_per_task=self.classes_list,
            feat_dim=self.in_channels[0],
            init='sphere'
        )       

        # 替换为简单头
        self.res_model = SimpleCosineSimHead(self.in_channels[0])

    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True

    def get_param_groups(self):
        param_groups = [[], []] 
        for param in list(self.encoder.parameters()):
            param_groups[0].append(param)

        for prototype in self.prototype_module.prototype:
            param_groups[1].append(prototype)
        
        for param in list(self.res_model.parameters()):
            param_groups[1].append(param)

        return param_groups

    def to_2D(self, x, h, w):
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self, x):
        cls_token, _x, x_aux = self.encoder.forward_features(x)
        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size
        _x4 = self.to_2D(_x, h, w)  # [B, F, H, W]
        
        P = self.prototype_module()   # [C_total, F]
        seg_proto, delta_p ,p_final = self.res_model(_x4, P)
        
        # 返回 6 个值，保持接口一致性
        return _x4, seg_proto, P, delta_p, p_final, None



