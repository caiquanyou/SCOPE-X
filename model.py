"""SCOPE-X V3.5.0 with Group Compression (Raw Feature Space)

This file merges the RandomGroupCodec sequence compression from grouping/
into the V3.5.0 multi-task, dual-CLS architecture.

Key features:
- Dual CLS tokens (RNA_CLS, ATAC_CLS) for contrastive learning
- Multi-task decoder (value, cluster, chr, modality)
- Optional group compression in RAW feature space [B, L, 4096] -> [B, gc, 4096]
- Peak status prediction using ALL ATAC tokens (not just masked ones)
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pathlib import Path
import os
from contextlib import redirect_stdout
import torch.distributed as dist
import numpy as np
import math
from typing import Optional, Tuple


# ============================================================
# Group Compression Components (from grouping/model.py)
# ============================================================

def _ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def _make_perm_indices(seq_len: int, generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate permutation and inverse permutation indices."""
    perm = torch.randperm(seq_len, generator=generator)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(seq_len, device=perm.device)
    return perm, inv


def _group_params(seq_len: int, num_groups: int) -> Tuple[int, int]:
    """Compute group size and padded length."""
    num_groups = max(1, int(num_groups))
    s = _ceil_div(seq_len, num_groups)
    gc = _ceil_div(seq_len, s)
    padded = gc * s
    return s, padded


def _build_node_mask(seq_len: int, gc: int, s: int, device, dtype=torch.bool) -> torch.Tensor:
    """Build boolean mask for valid nodes in groups."""
    total = gc * s
    flat = torch.arange(total, device=device) < seq_len
    return flat.view(gc, s).to(dtype)


class GroupDown(nn.Module):
    """Compress each group of s tokens to 1 token via linear/GCN/GAT."""

    def __init__(
        self,
        group_size: int,
        input_dim: int,
        mode: str = "linear",
        gcn_hidden: Optional[int] = None,
        gat_heads: int = 4,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.s = group_size
        self.d = input_dim
        gcn_hidden = gcn_hidden or min(256, input_dim)

        if self.mode == "linear":
            self.down = nn.Linear(self.s * self.d, self.d)
        elif self.mode == "gcn":
            self.gc_w1 = nn.Linear(self.d, gcn_hidden)
            self.gc_w2 = nn.Linear(gcn_hidden, self.d)
        elif self.mode == "gat":
            if self.d % gat_heads != 0:
                raise ValueError("input_dim must be divisible by gat_heads")
            self.heads = gat_heads
            self.d_head = self.d // gat_heads
            self.q = nn.Linear(self.d, self.d)
            self.k = nn.Linear(self.d, self.d)
            self.v = nn.Linear(self.d, self.d)
            self.gat_out = nn.Linear(self.d, self.d)
        else:
            raise ValueError("mode must be one of: linear, gcn, gat")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, gc, s, d = x.shape
        if mask.dim() == 2:
            m = mask.unsqueeze(0).expand(b, -1, -1)
        else:
            m = mask

        if self.mode == "linear":
            xm = x * m.unsqueeze(-1).to(x.dtype)
            flat = xm.reshape(b, gc, s * d)
            return self.down(flat)

        if self.mode == "gcn":
            m_f = m.to(x.dtype)
            adj = m_f.unsqueeze(-1) * m_f.unsqueeze(-2)
            eye = torch.eye(s, device=x.device, dtype=x.dtype).view(1, 1, s, s)
            adj = adj + eye * m_f.unsqueeze(-1)
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            a_norm = adj / deg
            h = x
            h = torch.matmul(a_norm, h)
            h = F.relu(self.gc_w1(h))
            h = torch.matmul(a_norm, h)
            h = self.gc_w2(h)
            denom = m_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            out = (h * m_f.unsqueeze(-1)).sum(dim=-2) / denom.squeeze(-1).unsqueeze(-1)
            return out

        # GAT mode
        m_f = m.to(x.dtype)
        h = x
        q = self.q(h).view(b, gc, s, self.heads, self.d_head).transpose(2, 3)
        k = self.k(h).view(b, gc, s, self.heads, self.d_head).transpose(2, 3)
        v = self.v(h).view(b, gc, s, self.heads, self.d_head).transpose(2, 3)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        neg = torch.finfo(scores.dtype).min
        valid = m_f.unsqueeze(1).unsqueeze(2) * m_f.unsqueeze(1).unsqueeze(-1)
        scores = scores.masked_fill(valid < 0.5, neg)
        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(torch.isnan(attn), 0.0)
        ctx = attn @ v
        ctx = ctx.transpose(2, 3).contiguous().view(b, gc, s, self.d)
        h2 = self.gat_out(ctx)
        denom = m_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
        out = (h2 * m_f.unsqueeze(-1)).sum(dim=-2) / denom.squeeze(-1).unsqueeze(-1)
        return out


class GroupUp(nn.Module):
    """Expand one compressed token back to s tokens."""

    def __init__(self, group_size: int, input_dim: int):
        super().__init__()
        self.s = group_size
        self.d = input_dim
        self.up = nn.Linear(self.d, self.s * self.d)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b, gc, d = z.shape
        flat = self.up(z)
        return flat.view(b, gc, self.s, d)


class RandomGroupCodec(nn.Module):
    """Random permutation + grouped sequence compression/expansion.

    Operates on raw feature space (input_dim = token feature dim, e.g., 4096).
    compress: [B, L, D] -> [B, gc, D]
    expand_embed: [B, gc, E] -> [B, L, E]  (for transformer output)
    """

    def __init__(
        self,
        input_dim: int,
        l_rna: int,
        l_atac: int,
        group_mode: str = "merged",
        num_groups: int = 32,
        down_mode: str = "linear",
        gat_heads: int = 4,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.l_rna = int(l_rna)
        self.l_atac = int(l_atac)
        self.group_mode = group_mode.lower()
        self.num_groups = int(num_groups)
        self.down_mode = down_mode.lower()

        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))

        if self.group_mode == "merged":
            self.l_total = self.l_rna + self.l_atac
            self.s, self.padded_len = _group_params(self.l_total, self.num_groups)
            self.gc = self.padded_len // self.s
            p, inv = _make_perm_indices(self.l_total, gen)
            self.register_buffer("perm_indices", p)
            self.register_buffer("inv_perm_indices", inv)
            self.register_buffer("node_mask", _build_node_mask(self.l_total, self.gc, self.s, p.device))
            self.down = GroupDown(self.s, input_dim, mode=self.down_mode, gat_heads=gat_heads)
        elif self.group_mode == "separate":
            if self.num_groups % 2 != 0:
                raise ValueError("num_groups must be even when group_mode is separate")
            half = self.num_groups // 2
            self.l_total = self.l_rna + self.l_atac
            self.s_rna, pad_r = _group_params(self.l_rna, half)
            self.gc_rna = pad_r // self.s_rna
            self.s_atac, pad_a = _group_params(self.l_atac, half)
            self.gc_atac = pad_a // self.s_atac
            self.gc = self.gc_rna + self.gc_atac

            pr, invr = _make_perm_indices(self.l_rna, gen)
            pa, inva = _make_perm_indices(self.l_atac, gen)
            self.register_buffer("perm_rna", pr)
            self.register_buffer("inv_rna", invr)
            self.register_buffer("perm_atac", pa)
            self.register_buffer("inv_atac", inva)
            self.register_buffer("mask_rna", _build_node_mask(self.l_rna, self.gc_rna, self.s_rna, pr.device))
            self.register_buffer("mask_atac", _build_node_mask(self.l_atac, self.gc_atac, self.s_atac, pa.device))

            self.down_rna = GroupDown(self.s_rna, input_dim, mode=self.down_mode, gat_heads=gat_heads)
            self.down_atac = GroupDown(self.s_atac, input_dim, mode=self.down_mode, gat_heads=gat_heads)
        else:
            raise ValueError("group_mode must be merged or separate")

    def compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress raw feature sequence [B, L, D] -> [B, gc, D]."""
        if self.group_mode == "merged":
            return self._compress_merged(x)
        return self._compress_separate(x)

    def expand_embed(self, z: torch.Tensor) -> torch.Tensor:
        """Expand compressed embeddings back to full sequence length.

        Args:
            z: [B, gc, embed_dim] - compressed embeddings from transformer

        Returns:
            [B, L, embed_dim] - expanded embeddings for decoding
        """
        if self.group_mode == "merged":
            b, gc, e = z.shape
            if gc != self.gc:
                raise ValueError(f"expected {self.gc} groups, got {gc}")

            # Create temporary GroupUp layer (operates on embed_dim)
            up_layer = GroupUp(group_size=self.s, input_dim=e).to(z.device)
            z_expanded = up_layer(z)  # [B, gc, s, E]
            z_flat = z_expanded.reshape(b, self.gc * self.s, e)  # [B, padded_len, E]

            # Unpermute
            inv_idx = self.inv_perm_indices.view(1, -1, 1).expand(b, self.padded_len, e)
            z_unperm = torch.gather(z_flat, 1, inv_idx)
            return z_unperm[:, :self.l_total, :]
        else:
            # Separate mode
            b, gc, e = z.shape
            expected_gc = self.gc_rna + self.gc_atac
            if gc != expected_gc:
                raise ValueError(f"expected {expected_gc} groups (separate), got {gc}")

            # Split RNA and ATAC parts
            z_rna = z[:, :self.gc_rna, :]   # [B, gc_rna, E]
            z_atac = z[:, self.gc_rna:, :]  # [B, gc_atac, E]

            # Expand each part
            up_rna = GroupUp(group_size=self.s_rna, input_dim=e).to(z.device)
            up_atac = GroupUp(group_size=self.s_atac, input_dim=e).to(z.device)

            z_rna_exp = up_rna(z_rna)  # [B, gc_rna, s_rna, E]
            z_atac_exp = up_atac(z_atac)  # [B, gc_atac, s_atac, E]

            z_rna_flat = z_rna_exp.reshape(b, self.gc_rna * self.s_rna, e)
            z_atac_flat = z_atac_exp.reshape(b, self.gc_atac * self.s_atac, e)

            # Unpermute
            inv_rna_idx = self.inv_rna.view(1, -1, 1).expand(b, self.gc_rna * self.s_rna, e)
            inv_atac_idx = self.inv_atac.view(1, -1, 1).expand(b, self.gc_atac * self.s_atac, e)

            z_rna_unperm = torch.gather(z_rna_flat, 1, inv_rna_idx)
            z_atac_unperm = torch.gather(z_atac_flat, 1, inv_atac_idx)

            # Trim padding and concatenate
            z_rna_final = z_rna_unperm[:, :self.l_rna, :]
            z_atac_final = z_atac_unperm[:, :self.l_atac, :]

            return torch.cat([z_rna_final, z_atac_final], dim=1)

    def _gather_perm(self, x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        idx = perm.view(1, l, 1).expand(b, l, d)
        return torch.gather(x, 1, idx)

    def _compress_merged(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        if l != self.l_total:
            raise ValueError(f"expected sequence length {self.l_total}, got {l}")
        x_p = self._gather_perm(x, self.perm_indices)
        pad = self.padded_len - l
        if pad > 0:
            x_p = F.pad(x_p, (0, 0, 0, pad))
        xg = x_p.view(b, self.gc, self.s, d)
        return self.down(xg, self.node_mask)

    def _compress_separate(self, x: torch.Tensor) -> torch.Tensor:
        b, l, d = x.shape
        if l != self.l_total:
            raise ValueError(f"expected sequence length {self.l_total}, got {l}")
        xr = x[:, : self.l_rna, :]
        xa = x[:, self.l_rna :, :]
        xr_p = self._gather_perm(xr, self.perm_rna)
        xa_p = self._gather_perm(xa, self.perm_atac)
        pad_r = self.gc_rna * self.s_rna - self.l_rna
        pad_a = self.gc_atac * self.s_atac - self.l_atac
        if pad_r > 0:
            xr_p = F.pad(xr_p, (0, 0, 0, pad_r))
        if pad_a > 0:
            xa_p = F.pad(xa_p, (0, 0, 0, pad_a))
        xgr = xr_p.view(b, self.gc_rna, self.s_rna, d)
        xga = xa_p.view(b, self.gc_atac, self.s_atac, d)
        zr = self.down_rna(xgr, self.mask_rna)
        za = self.down_atac(xga, self.mask_atac)
        return torch.cat([zr, za], dim=1)


# ============================================================
# V3.5.0 Core Architecture (with group compression integrated)
# ============================================================

def compute_contrastive_loss(proj_rna, proj_atac, temperature=0.07):
    """InfoNCE loss between RNA and ATAC CLS token projections."""
    proj_rna = F.normalize(proj_rna, dim=1)
    proj_atac = F.normalize(proj_atac, dim=1)
    logits = torch.matmul(proj_rna, proj_atac.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    loss_r2a = F.cross_entropy(logits, labels)
    loss_a2r = F.cross_entropy(logits.T, labels)
    return (loss_r2a + loss_a2r) / 2


class MultiDecoder(nn.Module):
    """Multi-task decoder heads."""

    def __init__(self, embed_dim, cluster_size, chr_size, modality_size, hidden_dim=256):
        super(MultiDecoder, self).__init__()

        def make_head(out_dim):
            return nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, out_dim)
            )

        self.value_head    = nn.Linear(embed_dim, 1)
        self.cluster_head  = make_head(cluster_size)
        self.chr_head      = make_head(chr_size)
        self.modality_head = make_head(modality_size)

    def forward(self, x):
        return {
            'value':    self.value_head(x),
            'cluster':  self.cluster_head(x),
            'chr':      self.chr_head(x),
            'modality': self.modality_head(x)
        }


class BERTForSelfSupervised(nn.Module):
    """V3.5.0 model with optional RandomGroupCodec sequence compression.

    When use_group_compress=True, the raw token feature sequence [B, L, input_dim]
    is first compressed to [B, gc, input_dim] via RandomGroupCodec before embedding.
    The transformer then operates on the shorter compressed sequence.
    After the transformer, the decoder output is expanded back to [B, L, embed_dim]
    via group_codec.expand_embed() for multi-task decoding.

    When use_group_compress=False (default), behavior is identical to V3.5.0.
    """

    def __init__(self, input_dim=4096, embed_dim=512, num_layers=6, MLP_hidden=64, nhead=8, dim_feedforward=1024,
                 id_input_dim=1, cluster_input_dim=4096, chr_input_dim=26, modality_input_dim=2,
                 rank_input_dim=5000,
                 # Group compression params (all optional)
                 use_group_compress: bool = False,
                 group_mode: str = "merged",
                 num_groups: int = 32,
                 down_mode: str = "linear",
                 gat_heads: int = 4,
                 perm_seed: Optional[int] = 42,
                 l_rna: int = 1000,
                 l_atac: int = 2000):
        super().__init__()
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.use_group_compress = use_group_compress

        # --- Group compression (raw feature space) ---
        if use_group_compress:
            self.group_codec = RandomGroupCodec(
                input_dim=input_dim,   # raw feature dim (e.g., 4096)
                l_rna=l_rna,
                l_atac=l_atac,
                group_mode=group_mode,
                num_groups=num_groups,
                down_mode=down_mode,
                gat_heads=gat_heads,
                seed=perm_seed,
            )

        # --- 6-channel embeddings (summed) ---
        self.id_embedding       = nn.Embedding(id_input_dim, embed_dim)
        self.value_embedding    = nn.Linear(1, embed_dim)
        self.cluster_embedding  = nn.Embedding(cluster_input_dim, embed_dim)
        self.chr_embedding      = nn.Embedding(chr_input_dim, embed_dim)
        self.modality_embedding = nn.Embedding(modality_input_dim, embed_dim)
        self.rank_embedding     = nn.Embedding(rank_input_dim, embed_dim)

        # --- ATAC open/close prediction head ---
        self.atac_status_head = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        # --- Dual CLS tokens ---
        self.cls_token_rna  = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cls_token_atac = nn.Parameter(torch.randn(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # --- Contrastive learning projection head ---
        self.sim_projection = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

        # --- Multi-task decoder ---
        self.decoder = MultiDecoder(
            embed_dim=self.embed_dim,
            cluster_size=cluster_input_dim,
            chr_size=chr_input_dim,
            modality_size=modality_input_dim
        )

    def predict_atac_from_cls(self, cls_atac, peak_ids):
        """Predict peak open/closed status using CLS vector + peak ID embedding."""
        peak_embed = self.id_embedding(peak_ids)  # [N, E]
        combined_feature = torch.cat([cls_atac, peak_embed], dim=-1)  # [N, 2E]
        logits = self.atac_status_head(combined_feature)  # [N, 1]
        return logits

    def forward(self, x):
        batch, length, feature = x.shape

        # --- Optional: compress raw token sequence BEFORE embedding ---
        if self.use_group_compress:
            x = self.group_codec.compress(x)  # [B, gc, input_dim]
            batch, length, feature = x.shape  # length = gc now

        # --- Extract 6 feature channels and embed ---
        token_ids    = x[:, :, 0].long()
        values       = x[:, :, 1].float().unsqueeze(-1)
        chr_ids      = x[:, :, 2].long()
        cluster_ids  = x[:, :, 3].long()
        modality_ids = x[:, :, 4].long()
        rank_ids     = x[:, :, 5].long()

        id_feature       = self.id_embedding(token_ids)
        value_feature    = self.value_embedding(values)
        cluster_feature  = self.cluster_embedding(cluster_ids)
        chr_feature      = self.chr_embedding(chr_ids)
        modality_feature = self.modality_embedding(modality_ids)
        rank_feature     = self.rank_embedding(rank_ids)

        x_emb = id_feature + value_feature + cluster_feature + chr_feature + modality_feature + rank_feature
        # [B, gc, E] or [B, L, E]

        # --- Prepend RNA_CLS, append ATAC_CLS ---
        cls_rna  = self.cls_token_rna.expand(batch, -1, -1)
        cls_atac = self.cls_token_atac.expand(batch, -1, -1)
        x_emb = torch.cat([cls_rna, x_emb, cls_atac], dim=1)  # [B, gc+2, E] or [B, L+2, E]

        total_len = x_emb.size(1)
        attn_mask = torch.zeros((batch, total_len, total_len), dtype=torch.bool, device=x_emb.device)

        if self.use_group_compress:
            # After compression, tokens are mixed groups — only block CLS cross-attention
            attn_mask[:, 0, -1] = True   # RNA_CLS cannot attend to ATAC_CLS
            attn_mask[:, -1, 0] = True   # ATAC_CLS cannot attend to RNA_CLS
        else:
            # Full per-token modality masking (V3.5.0 original behavior)
            modality_flat = modality_ids.reshape(batch, -1)
            is_rna_data  = (modality_flat == 0)
            is_atac_data = (modality_flat == 1)
            attn_mask[:, 0, -1]    = True
            attn_mask[:, 0, 1:-1]  = is_atac_data
            attn_mask[:, -1, 0]    = True
            attn_mask[:, -1, 1:-1] = is_rna_data

        attn_mask = attn_mask.repeat_interleave(self.nhead, dim=0)

        encoded = self.transformer(x_emb, mask=attn_mask)  # [B, gc+2, E] or [B, L+2, E]

        out_cls_rna  = encoded[:, 0, :]    # [B, E]
        out_cls_atac = encoded[:, -1, :]   # [B, E]
        out_data     = encoded[:, 1:-1, :] # [B, gc, E] or [B, L, E]

        # --- Expand back to full sequence length for decoding ---
        if self.use_group_compress:
            out_data = self.group_codec.expand_embed(out_data)  # [B, gc, E] -> [B, L, E]

        mlm_logits = self.decoder(out_data)

        proj_rna  = self.sim_projection(out_cls_rna)
        proj_atac = self.sim_projection(out_cls_atac)

        return mlm_logits, proj_rna, proj_atac, out_cls_rna, out_cls_atac


def compute_multitask_loss(decoded, targets, mask, weights=None):
    """Compute multi-task loss (value, cluster, chr, modality)."""
    if weights is None:
        weights = {'value': 1.0, 'cluster': 1.0, 'chr': 1.0, 'modality': 1.0}

    total_loss = 0.0
    loss_dict = {}

    loss_val = F.mse_loss(decoded['value'][mask].squeeze(-1), targets['value'][mask])
    w_val = weights.get('value', 1.0)
    total_loss += w_val * loss_val
    loss_dict['loss_value'] = loss_val.item()

    loss_cluster = F.cross_entropy(decoded['cluster'][mask], targets['cluster'][mask])
    w_cluster = weights.get('cluster', 1.0)
    total_loss += w_cluster * loss_cluster
    loss_dict['loss_cluster'] = loss_cluster.item()

    loss_chr = F.cross_entropy(decoded['chr'][mask], targets['chr'][mask])
    w_chr = weights.get('chr', 1.0)
    total_loss += w_chr * loss_chr
    loss_dict['loss_chr'] = loss_chr.item()

    loss_modality = F.cross_entropy(decoded['modality'][mask], targets['modality'][mask])
    w_modality = weights.get('modality', 1.0)
    total_loss += w_modality * loss_modality
    loss_dict['loss_modality'] = loss_modality.item()

    individual_loss=[loss_val,loss_cluster,loss_chr,loss_modality]
    return total_loss, loss_dict


def train_model(model, dataloader, optimizer, config, save_dir, rank):
    """Training loop with fixed peak status prediction (use ALL ATAC tokens)."""
    model.train()

    loss_weights = config.get('loss_weights', {
        'value': 1.0,
        'cluster': 1.0,
        'chr': 0.0,
        'modality': 0.0
    })
    alpha_contra = config.get('alpha_contra', 0)
    w_atac = loss_weights.get('atac_bce', 1.0)

    bce_criterion = nn.BCEWithLogitsLoss()

    train_loss=[]
    train_precision=[]
    train_recall=[]
    train_f1=[]
    train_ATAC_acc=[]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for epoch in range(config['num_epochs']):
        epoch_loss = 0
        epoch_atac_acc_sum = 0.0
        eopch_precision=0
        epoch_recall=0
        epoch_f1=0
        epoch_atac_steps = 0

        epoch_task_rna_sum = 0.0
        epoch_bce_sum = 0.0
        epoch_loss_value_sum   = 0.0
        epoch_loss_cluster_sum = 0.0
        epoch_loss_chr_sum     = 0.0
        epoch_loss_mod_sum     = 0.0

        for masked_input, target, loss_mask in dataloader:
            masked_input = masked_input.float().cuda(rank, non_blocking=True)
            target = target.float().cuda(rank, non_blocking=True)
            loss_mask = loss_mask.bool().cuda(rank, non_blocking=True)

            target_dict = {
                'id':        target[:, :, 0].long(),
                'value':     target[:, :, 1].float(),
                'chr':       target[:, :, 2].long(),
                'cluster':   target[:, :, 3].long(),
                'modality':  target[:, :, 4].long()}

            optimizer.zero_grad()

            output, proj_rna, proj_atac, out_cls_rna, out_cls_atac = model(masked_input)

            # --- ✅ FIXED: Use ALL ATAC tokens for BCE loss (not just masked ones) ---
            # Peak status prediction is a discriminative task independent of MLM masking
            is_atac = (target_dict['modality'] == 1)  # Use all ATAC tokens

            loss_atac_bce_atac = torch.tensor(0.0, device=masked_input.device)
            loss_atac_bce_rna = torch.tensor(0.0, device=masked_input.device)

            if is_atac.any():
                target_vals = target_dict['value'][is_atac]
                target_labels = (target_vals > 0).float().unsqueeze(-1)
                target_peak_ids = target_dict['id'][is_atac]

                batch_indices, _ = torch.nonzero(is_atac, as_tuple=True)

                # ATAC CLS -> Peak
                gathered_cls_atac = out_cls_atac[batch_indices]
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    logits_atac = model.module.predict_atac_from_cls(gathered_cls_atac, target_peak_ids)
                else:
                    logits_atac = model.predict_atac_from_cls(gathered_cls_atac, target_peak_ids)
                loss_atac_bce_atac = bce_criterion(logits_atac, target_labels)

                # RNA CLS -> Peak
                gathered_cls_rna = out_cls_rna[batch_indices]
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    logits_rna = model.module.predict_atac_from_cls(gathered_cls_rna, target_peak_ids)
                else:
                    logits_rna = model.predict_atac_from_cls(gathered_cls_rna, target_peak_ids)
                loss_atac_bce_rna = bce_criterion(logits_rna, target_labels)

                # Compute Accuracy (based on ATAC CLS)
                with torch.no_grad():
                    preds = (torch.sigmoid(logits_atac) > 0.5).float()
                    correct = (preds == target_labels).sum().item()

                    TP = (preds  * target_labels).sum()
                    FP = (preds  * (1-target_labels.int())).sum()
                    FN = ((1-preds .int()) * target_labels).sum()

                    eps = torch.finfo(torch.float32).eps
                    precision_batch = TP.float() / (TP.float() + FP.float() + eps)
                    recall_batch = TP.float() / (TP.float() + FN.float() + eps)
                    f1_batch = 2 * (precision_batch * recall_batch) / (precision_batch + recall_batch + eps)

                    eopch_precision+=precision_batch
                    epoch_recall+=recall_batch
                    epoch_f1+=f1_batch

                    batch_atac_acc = correct / target_labels.size(0)
                    epoch_atac_acc_sum += batch_atac_acc
                    epoch_atac_steps += 1

            # Flatten data for MLM Loss
            for key in output:
                output[key] = output[key].reshape(-1, output[key].shape[-1])
            for key in target_dict:
                target_dict[key] = target_dict[key].reshape(-1)

            mask_flat = loss_mask.view(-1)
            modality_flat = target_dict['modality']

            # Only calculate RNA area MLM
            is_rna_mask_flat = mask_flat & (modality_flat == 0)

            task_loss_rna, loss_parts = compute_multitask_loss(output, target_dict, is_rna_mask_flat, weights=loss_weights)

            epoch_loss_value_sum   += loss_parts["loss_value"]
            epoch_loss_cluster_sum += loss_parts["loss_cluster"]
            epoch_loss_chr_sum     += loss_parts["loss_chr"]
            epoch_loss_mod_sum     += loss_parts["loss_modality"]

            # Contrastive Loss
            contra_loss = compute_contrastive_loss(proj_rna, proj_atac)

            # Total Loss
            total_loss = task_loss_rna + w_atac * (loss_atac_bce_atac + loss_atac_bce_rna) + alpha_contra * contra_loss

            # Accumulate stats
            bce_sum = (loss_atac_bce_atac + loss_atac_bce_rna)
            epoch_task_rna_sum += task_loss_rna.detach().item()
            epoch_bce_sum += bce_sum.detach().item()

            torch.distributed.all_reduce(total_loss, op=torch.distributed.ReduceOp.AVG)
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()

        avg_loss = epoch_loss / len(dataloader)
        train_loss.append(avg_loss)
        avg_precision = eopch_precision / epoch_atac_steps if epoch_atac_steps > 0 else 0.0
        avg_recall = epoch_recall / epoch_atac_steps if epoch_atac_steps > 0 else 0.0
        avg_f1 = epoch_f1 / epoch_atac_steps if epoch_atac_steps > 0 else 0.0
        avg_atac_acc = epoch_atac_acc_sum / epoch_atac_steps if epoch_atac_steps > 0 else 0.0

        train_precision.append(avg_precision.item())
        train_recall.append(avg_recall.item())
        train_f1.append(avg_f1.item())
        train_ATAC_acc.append(avg_atac_acc)

        torch.distributed.barrier()

        if rank == 0:
            avg_task_rna = epoch_task_rna_sum / len(dataloader)
            avg_bce_sum  = epoch_bce_sum / len(dataloader)
            avg_lv   = epoch_loss_value_sum   / len(dataloader)
            avg_lc   = epoch_loss_cluster_sum / len(dataloader)
            avg_lchr = epoch_loss_chr_sum     / len(dataloader)
            avg_lm   = epoch_loss_mod_sum     / len(dataloader)

            weighted_bce = w_atac * avg_bce_sum
            ratio = weighted_bce / (avg_task_rna + 1e-12)

            print(
                f"Epoch {epoch + 1}/{config['num_epochs']} | "
                f"total={avg_loss:.4f} | "
                f"task_rna={avg_task_rna:.4f} | "
                f"mlm(value={avg_lv:.4f}, cluster={avg_lc:.4f}, chr={avg_lchr:.4f}, mod={avg_lm:.4f}) | "
                f"bce_sum={avg_bce_sum:.4f} | "
                f"w*bce={weighted_bce:.4f} | "
                f"ratio(w*bce/task)={ratio:.2f} | "
                f"ATAC Acc={avg_atac_acc:.4f} P={avg_precision:.4f} R={avg_recall:.4f} F1={avg_f1:.4f}",
                flush=True
            )

            model_name = f"{config['sample_num']}_sample_model_epoch{epoch+1}_loss{avg_loss:.4f}.pth"
            save_path = save_dir / model_name
            torch.save(model.module.state_dict(), save_path)

    return train_loss,[train_precision,train_recall,train_f1,train_ATAC_acc]


def evaluate_model_DP(model, test_loader):
    """Evaluation function (DP version)."""
    model.eval()
    all_rna_cls = []
    all_atac_cls = []

    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for masked_input, target, loss_mask in test_loader:
            masked_input = masked_input.float().cuda()
            target = target.float().cuda()
            loss_mask = loss_mask.bool().cuda()

            _, proj_rna, proj_atac, _, out_cls_atac = model(masked_input)

            target_dict = {
                'id': target[:, :, 0].long(),
                'value': target[:, :, 1].float(),
                'modality': target[:, :, 4].long()
            }
            loss_mask=True
            is_atac_mask = loss_mask & (target_dict['modality'] == 1)

            if is_atac_mask.any():
                target_vals = target_dict['value'][is_atac_mask]
                target_labels = (target_vals > 0).float().unsqueeze(-1)
                target_peak_ids = target_dict['id'][is_atac_mask]

                batch_indices, _ = torch.nonzero(is_atac_mask, as_tuple=True)
                gathered_cls_atac = out_cls_atac[batch_indices]

                logits_atac = model.predict_atac_from_cls(gathered_cls_atac, target_peak_ids)
                preds = (torch.sigmoid(logits_atac) > 0.5).float()
                total_correct += (preds == target_labels).sum().item()
                total_samples += target_labels.size(0)

            all_rna_cls.append(proj_rna.cpu())
            all_atac_cls.append(proj_atac.cpu())
            torch.cuda.empty_cache()

    all_rna_cls = torch.cat(all_rna_cls, dim=0).numpy()
    all_atac_cls = torch.cat(all_atac_cls, dim=0).numpy()
    combined_cls = np.concatenate([all_rna_cls, all_atac_cls], axis=1)
    print("combined_cls shape:", combined_cls.shape)

    avg_acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"Evaluation ATAC Accuracy: {avg_acc:.4f}")

    return all_rna_cls, all_atac_cls,combined_cls, avg_acc


def plot(train_loss, save_dir, show=True, save=True):
    """Plot training loss curve."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    if train_loss:
        max_loss = max(train_loss)
        plt.ylim(0, max_loss * 1.1)

    if save:
        plot_path = save_dir/"loss_curve.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_individual_result(train_result,save_dir,show=True, save=True):
    """Plot individual metrics."""
    plot_name=['precision','recall','f1','avg_acc']
    for i in range(len(plot_name)):
        plt.figure(figsize=(10, 6))

        epochs = range(1, len(train_result[i]) + 1)
        plt.plot(epochs, train_result[i], label=f'Training {plot_name[i]} ')

        plt.title(f"{plot_name[i]}")
        plt.xlabel('Epochs')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)

        max_loss = max(train_result[i])
        min_loss=min(train_result[i])
        plt.ylim(min_loss*0.9, max_loss * 1.1)

        if save:
            plot_path = save_dir/f"result_{plot_name[i]}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
