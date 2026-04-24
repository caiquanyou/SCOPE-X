import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalTranslatorHead(nn.Module):
    """Token-level translator head for RNA->ATAC or ATAC->RNA."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RepresentationClassifierHead(nn.Module):
    """Classification head on fused CLS embedding."""

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, cls_rna: torch.Tensor, cls_atac: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.cat([cls_rna, cls_atac], dim=-1))


class RetrievalProjectionHead(nn.Module):
    """Projection head for cross-modal retrieval."""

    def __init__(self, embed_dim: int, proj_dim: int = 128):
        super().__init__()
        self.rna = nn.Linear(embed_dim, proj_dim)
        self.atac = nn.Linear(embed_dim, proj_dim)

    def forward(self, cls_rna: torch.Tensor, cls_atac: torch.Tensor):
        z_rna = F.normalize(self.rna(cls_rna), dim=-1)
        z_atac = F.normalize(self.atac(cls_atac), dim=-1)
        return z_rna, z_atac


class PeakGeneLinkHead(nn.Module):
    """Predict peak-gene link logits from pairwise token embeddings."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, peak_embed: torch.Tensor, gene_embed: torch.Tensor, distance: torch.Tensor):
        if distance.dim() == 1:
            distance = distance.unsqueeze(-1)
        x = torch.cat([peak_embed, gene_embed, distance], dim=-1)
        return self.mlp(x)


class PerturbationResponseHead(nn.Module):
    """Predict perturbation response vectors from fused CLS embedding."""

    def __init__(self, embed_dim: int, out_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, cls_rna: torch.Tensor, cls_atac: torch.Tensor):
        x = torch.cat([cls_rna, cls_atac], dim=-1)
        return self.net(x)
