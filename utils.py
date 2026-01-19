import torch 

def confusion_matrix_gpu(preds, labels, num_classes):
    """在GPU上计算混淆矩阵，等价于 sklearn.metrics.confusion_matrix(labels, preds)。"""
    preds = preds.long()
    labels = labels.long()
    device = preds.device
    # 展开为一维直方图：index = label * num_classes + pred
    cm = torch.zeros(num_classes * num_classes, device=device, dtype=torch.long)
    indices = labels * num_classes + preds
    ones = torch.ones_like(indices, dtype=torch.long)
    cm.scatter_add_(0, indices, ones)
    cm = cm.view(num_classes, num_classes)
    return cm


def kappa_coefficient_gpu(cm):
    """根据混淆矩阵在GPU上计算 Cohen's kappa 系数，返回 Python float。"""
    cm = cm.float()
    total = cm.sum()
    if total == 0:
        return 0.0
    pa = torch.trace(cm) / total
    row_sum = cm.sum(dim=1)
    col_sum = cm.sum(dim=0)
    pe = (row_sum * col_sum).sum() / (total * total)
    if pe == 1.0:
        return 0.0
    kappa = (pa - pe) / (1.0 - pe)
    return float(kappa.item())


def tpr_gpu(cm):
    """在GPU上根据混淆矩阵计算每一类的 TPR，返回 numpy 数组。"""
    cm = cm.float()
    tp = torch.diag(cm)
    fn = cm.sum(dim=1) - tp
    denom = tp + fn
    tpr = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    return tpr.cpu().numpy()