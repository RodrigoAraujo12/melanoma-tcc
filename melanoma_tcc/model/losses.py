import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma: float = 2.0,
                 label_smoothing: float = 0.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        num_classes = logits.size(-1)

        if self.label_smoothing > 0:
            target_smooth = torch.full_like(log_probs, self.label_smoothing / num_classes)
            target_smooth.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing + self.label_smoothing / num_classes)
            ce = -(target_smooth * log_probs).sum(dim=-1)
            pt = (probs * target_smooth).sum(dim=-1).clamp(min=1e-8, max=1.0)
        else:
            ce = F.nll_loss(log_probs, target, reduction="none")
            pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[target]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(class_counts: list, mode: str = "inverse") -> torch.Tensor:
    counts = torch.tensor(class_counts, dtype=torch.float32)
    if mode == "inverse":
        weights = 1.0 / counts
    elif mode == "inverse_sqrt":
        weights = 1.0 / counts.sqrt()
    elif mode == "effective":
        beta = 0.999
        weights = (1.0 - beta) / (1.0 - beta ** counts)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    weights = weights / weights.sum() * len(counts)
    return weights
