import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha_mode='adaptive', num_classes=5, device='cuda'):
        super().__init__()
        self.gamma = gamma
        self.alpha_mode = alpha_mode
        self.num_classes = num_classes
        self.device = device
        
        # 初始化alpha为全1（后续会更新）
        self.register_buffer('alpha', torch.ones(num_classes, device=device))
        
        if alpha_mode == 'balanced':
            self.alpha.fill_(1.0 / num_classes)

    def update_alpha(self, class_counts):
        if self.alpha_mode == 'adaptive' and class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float32, device=self.device)
            self.alpha = (1.0 / (counts + 1e-6))  # 逆频率加权
            self.alpha = self.alpha / self.alpha.sum() * self.num_classes  # 归一化

    def forward(self, inputs, targets):
        # 确保targets是长整型
        targets = targets.long()
        
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # 计算focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        # 应用类别权重
        if self.alpha is not None:
            # 安全索引 - 确保targets值在有效范围内
            valid_mask = (targets >= 0) & (targets < self.num_classes)
            targets = targets[valid_mask]
            focal_loss = focal_loss[valid_mask]
            
            if len(targets) > 0:
                alpha_t = self.alpha[targets]
                focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean() if len(focal_loss) > 0 else torch.tensor(0.0, device=self.device)