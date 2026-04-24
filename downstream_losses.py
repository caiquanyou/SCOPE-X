from typing import Dict, Optional
import torch
import torch.nn.functional as F


def translation_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    is_binary: bool = False,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if is_binary:
        return F.binary_cross_entropy_with_logits(pred, target)
    return F.mse_loss(pred, target)


def cycle_consistency_loss(x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_recon, x_target)


def retrieval_infonce_loss(z_rna: torch.Tensor, z_atac: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    logits = torch.matmul(z_rna, z_atac.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_1 + loss_2)


def representation_cls_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels)


def link_prediction_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.float().view_as(logits)
    return F.binary_cross_entropy_with_logits(logits, labels)


def perturbation_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def weighted_sum(losses: Dict[str, torch.Tensor], weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
    if not losses:
        raise ValueError("losses must not be empty")
    if weights is None:
        return sum(losses.values())
    total = 0.0
    for k, v in losses.items():
        total = total + weights.get(k, 1.0) * v
    return total
