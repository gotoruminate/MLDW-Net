import torch
import torch.nn as nn

class MultiSimilarityLoss(nn.Module):
    """Multi-Similarity Loss for multi-class contrastive learning.
    Reference: "Multi-Similarity Loss with General Pair Weighting for Deep Metric Learning" (CVPR 2019).
    """
    def __init__(self, alpha=2.0, beta=50.0, lambda_pos=0.5, lambda_neg=1.0, 
                 temperature=0.07, device=None):
        super(MultiSimilarityLoss, self).__init__()
        self.alpha = alpha          # 正样本对权重系数
        self.beta = beta            # 负样本对权重系数
        self.lambda_pos = lambda_pos  # 正样本边界阈值
        self.lambda_neg = lambda_neg  # 负样本边界阈值
        self.temperature = temperature  # 温度系数（用于缩放相似度）
        self.device = device

    def forward(self, features, labels=None, mask=None, adv=True):
        """Compute Multi-Similarity Loss.
        
        Args:
            features: Hidden vectors [bsz, n_views, dim].
            labels: Ground truth labels [bsz].
            mask: Predefined mask [bsz, bsz] (optional).
        Returns:
            Loss scalar.
        """
        device = self.device if self.device is not None else features.device

        # 输入维度检查（与SupConLoss保持一致）
        if len(features.shape) < 3:
            raise ValueError('`features` must be [bsz, n_views, dim]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        
        # 生成标签mask（与SupConLoss逻辑一致）
        if labels is not None and mask is not None:
            raise ValueError('Cannot use both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Label size mismatch')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # 展开多视图特征（与SupConLoss一致）
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature  # 默认使用所有样本作为锚点（'all'模式）
        
        # 计算相似度矩阵（缩放后）
        sim_matrix = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        sim_matrix = torch.exp(sim_matrix - torch.max(sim_matrix, dim=1, keepdim=True)[0].detach())

        # 正负样本对mask
        pos_mask = mask.repeat(contrast_count, contrast_count)
        neg_mask = 1 - pos_mask
        logits_mask = torch.scatter(
            torch.ones_like(pos_mask),
            1,
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device),
            0
        )
        pos_mask = pos_mask * logits_mask  # 排除自对比
        neg_mask = neg_mask * logits_mask  # 排除自对比

        # --- Multi-Similarity 核心计算 ---
        # 1. 正样本损失（拉近同类样本）
        pos_sim = sim_matrix * pos_mask
        pos_loss = (1.0 / self.alpha) * torch.log(
            1 + torch.sum(pos_sim * torch.exp(-self.alpha * (pos_sim - self.lambda_pos)), dim=1)
        )

        # 2. 负样本损失（推开不同类样本）
        neg_sim = sim_matrix * neg_mask
        neg_loss = (1.0 / self.beta) * torch.log(
            1 + torch.sum(neg_sim * torch.exp(self.beta * (neg_sim - self.lambda_neg)), dim=1)
        )

        # 合并损失（按样本数平均）
        loss = torch.mean(pos_loss + neg_loss)
        
        return loss


# 测试代码（与SupConLoss接口一致）
if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(32, 2, 10)
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.randint(0, 10, [32])
    
    loss_fn = MultiSimilarityLoss(device='cpu')
    loss = loss_fn(x, y)
    print("Multi-Similarity Loss:", loss.item())