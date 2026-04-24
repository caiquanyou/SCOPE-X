"""Multi-task downstream test script aligned with pretraining data format.

This script reuses the same data pipeline as pretraining (`utils.load_multiple_samples` +
`utils.MaskedDataset`) and runs lightweight downstream-style evaluations:

1) RNA->ATAC translation proxy (mask ATAC tokens, predict ATAC value)
2) ATAC->RNA translation proxy (mask RNA tokens, predict RNA value)
3) Cross-modal retrieval (R@1 / R@5 from CLS projections)
4) Representation alignment (paired CLS cosine statistics)
5) ATAC open/close prediction accuracy (from ATAC_CLS head)

Expected model input format (same as pretraining forward):
[id, value, chr, cluster, modality, rank]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from model import BERTForSelfSupervised
from utils import MaskedDataset, load_multiple_samples, concatenate_samples, set_seed


def build_config(args) -> Dict:
    if args.config_json:
        with open(args.config_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "base_dir": args.base_dir,
            "ENSG2token_path": args.ENSG2token_path,
            "gene_cluster_info_path": args.gene_cluster_info_path,
            "gene_position_info_path": args.gene_position_info_path,
            "peak2token_path": args.peak2token_path,
            "peak_cluster_path": args.peak_cluster_path,
            "idf_path": args.idf_path,
            "loaded_dict": args.loaded_dict,
        }

    cfg.update(
        {
            "N_gene": args.N_gene,
            "N_peak": args.N_peak,
            "gene_position_range": args.gene_position_range,
            "max_pos_diff": args.max_pos_diff,
            "mask_ratio": args.mask_ratio,
            "batch_size": args.batch_size,
            "eval_cells": args.eval_cells,
            "embed_dim": args.embed_dim,
            "nhead": args.nhead,
            "num_layers": args.num_layers,
            "MLP_hidden": args.MLP_hidden,
            "dim_feedforward": args.dim_feedforward,
            "cluster_input_dim": args.cluster_input_dim,
            "chr_input_dim": args.chr_input_dim,
            "modality_input_dim": args.modality_input_dim,
            "rank_input_dim": args.rank_input_dim,
        }
    )
    return cfg


def load_sample_ids(loaded_dict_path: str, run_ids: List[str]) -> List[str]:
    with open(loaded_dict_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    sample_ids: List[str] = []
    for run_id in run_ids:
        samples = data.get(run_id, {}).get("samples", [])
        sample_ids.extend([x["sample"] for x in samples])
    return sample_ids


def build_eval_dataset(cfg: Dict, sample_ids: List[str], rank: int = 0):
    data_dict, *_ , vocab_size = load_multiple_samples(
        sample_ids,
        cfg["base_dir"],
        cfg["N_gene"],
        cfg["N_peak"],
        cfg["ENSG2token_path"],
        cfg["gene_position_info_path"],
        cfg["gene_cluster_info_path"],
        cfg["gene_position_range"],
        cfg["peak2token_path"],
        cfg["peak_cluster_path"],
        cfg["idf_path"],
        rank,
    )

    combined = concatenate_samples(data_dict)
    dataset = MaskedDataset(
        combined,
        masked_ratio=cfg["mask_ratio"],
        N_gene=cfg["N_gene"],
        N_peak=cfg["N_peak"],
        max_pos_diff=cfg["max_pos_diff"],
    )

    eval_cells = min(cfg["eval_cells"], len(dataset))
    subset = Subset(dataset, list(range(eval_cells)))
    return subset, int(vocab_size)


def build_model(cfg: Dict, vocab_size: int, device: torch.device) -> BERTForSelfSupervised:
    model = BERTForSelfSupervised(
        input_dim=6,
        embed_dim=cfg["embed_dim"],
        num_layers=cfg["num_layers"],
        MLP_hidden=cfg["MLP_hidden"],
        nhead=cfg["nhead"],
        dim_feedforward=cfg["dim_feedforward"],
        id_input_dim=vocab_size,
        cluster_input_dim=cfg["cluster_input_dim"],
        chr_input_dim=cfg["chr_input_dim"],
        modality_input_dim=cfg["modality_input_dim"],
        rank_input_dim=cfg["rank_input_dim"],
        use_group_compress=False,
        l_rna=cfg["N_gene"],
        l_atac=cfg["N_peak"],
    ).to(device)
    return model


def maybe_load_ckpt(model: torch.nn.Module, ckpt_path: str | None, device: torch.device):
    if not ckpt_path:
        return
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # handle DDP wrappers
    fixed = {}
    for k, v in state.items():
        nk = k.replace("module.", "")
        fixed[nk] = v
    model.load_state_dict(fixed, strict=False)


def make_cross_modal_input(
    target: torch.Tensor,
    n_gene: int,
    n_peak: int,
    mode: str,
) -> torch.Tensor:
    """Build conditional input by masking one modality.

    target format expected: [B, L, 6] = [id, value, chr, cluster, modality, rank]
    """
    x = target.clone()

    # keep modality + chr as structural hints, mask content channels
    content_cols = [0, 1, 3, 5]
    if mode == "rna_to_atac":
        x[:, n_gene : n_gene + n_peak, content_cols] = 0
    elif mode == "atac_to_rna":
        x[:, :n_gene, content_cols] = 0
    else:
        raise ValueError(f"unknown mode: {mode}")
    return x


def retrieval_metrics(z_rna: torch.Tensor, z_atac: torch.Tensor, topk=(1, 5)) -> Dict[str, float]:
    z_rna = torch.nn.functional.normalize(z_rna, dim=-1)
    z_atac = torch.nn.functional.normalize(z_atac, dim=-1)
    sim = z_rna @ z_atac.T

    labels = torch.arange(sim.shape[0], device=sim.device)
    metrics = {}
    for k in topk:
        pred = sim.topk(k, dim=1).indices
        hit = (pred == labels[:, None]).any(dim=1).float().mean().item()
        metrics[f"r_atac_from_rna@{k}"] = hit

        pred_rev = sim.T.topk(k, dim=1).indices
        hit_rev = (pred_rev == labels[:, None]).any(dim=1).float().mean().item()
        metrics[f"r_rna_from_atac@{k}"] = hit_rev
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, n_gene: int, n_peak: int):
    model.eval()

    mse_r2a_sum, mse_a2r_sum = 0.0, 0.0
    bce_acc_sum, steps = 0.0, 0

    all_proj_rna, all_proj_atac = [], []
    all_cls_cos = []

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    for masked_input, target, _ in loader:
        target = target.to(device).float()

        # --- Translation proxy: RNA->ATAC
        x_r2a = make_cross_modal_input(target, n_gene, n_peak, mode="rna_to_atac")
        out_r2a, _, _, _, cls_atac_r2a = model(x_r2a)
        pred_atac = out_r2a["value"][:, n_gene : n_gene + n_peak, 0]
        true_atac = target[:, n_gene : n_gene + n_peak, 1]
        mse_r2a_sum += torch.mean((pred_atac - true_atac) ** 2).item()

        # --- Translation proxy: ATAC->RNA
        x_a2r = make_cross_modal_input(target, n_gene, n_peak, mode="atac_to_rna")
        out_a2r, _, _, cls_rna_a2r, _ = model(x_a2r)
        pred_rna = out_a2r["value"][:, :n_gene, 0]
        true_rna = target[:, :n_gene, 1]
        mse_a2r_sum += torch.mean((pred_rna - true_rna) ** 2).item()

        # --- Retrieval + representation alignment (full input)
        _, proj_rna, proj_atac, cls_rna, cls_atac = model(target)
        all_proj_rna.append(proj_rna)
        all_proj_atac.append(proj_atac)
        cos = torch.nn.functional.cosine_similarity(cls_rna, cls_atac, dim=-1)
        all_cls_cos.append(cos)

        # --- ATAC status accuracy (same logic as pretraining head)
        is_atac = target[:, :, 4] == 1
        batch_idx, token_idx = torch.where(is_atac)
        if len(batch_idx) > 0:
            peak_ids = target[:, :, 0].long()[batch_idx, token_idx]
            cls_for_peak = cls_atac_r2a[batch_idx]
            logits = model.predict_atac_from_cls(cls_for_peak, peak_ids).squeeze(-1)

            labels = (target[:, :, 1][batch_idx, token_idx] > 0).float()
            _ = bce(logits, labels).mean().item()  # keeps interface consistent
            preds = (torch.sigmoid(logits) > 0.5).float()
            bce_acc_sum += (preds == labels).float().mean().item()

        steps += 1

    proj_rna = torch.cat(all_proj_rna, dim=0)
    proj_atac = torch.cat(all_proj_atac, dim=0)
    retrieval = retrieval_metrics(proj_rna, proj_atac, topk=(1, 5))

    out = {
        "translation_mse_rna_to_atac": mse_r2a_sum / max(steps, 1),
        "translation_mse_atac_to_rna": mse_a2r_sum / max(steps, 1),
        "repr_cls_cosine_mean": torch.cat(all_cls_cos).mean().item(),
        "atac_open_close_acc": bce_acc_sum / max(steps, 1),
    }
    out.update(retrieval)
    return out


def parse_args():
    p = argparse.ArgumentParser(description="SCOPE-X multi-downstream test script")

    # Optional full config json
    p.add_argument("--config_json", type=str, default=None)

    # Path args (same semantics as pretraining script)
    p.add_argument("--base_dir", type=str, default="")
    p.add_argument("--ENSG2token_path", type=str, default="")
    p.add_argument("--gene_cluster_info_path", type=str, default="")
    p.add_argument("--gene_position_info_path", type=str, default="")
    p.add_argument("--peak2token_path", type=str, default="")
    p.add_argument("--peak_cluster_path", type=str, default="")
    p.add_argument("--idf_path", type=str, default="")
    p.add_argument("--loaded_dict", type=str, default="")

    p.add_argument("--run_ids", type=str, default="run_1,run_2,run_3")

    # data/model args
    p.add_argument("--N_gene", type=int, default=1000)
    p.add_argument("--N_peak", type=int, default=2000)
    p.add_argument("--gene_position_range", type=int, default=500)
    p.add_argument("--max_pos_diff", type=int, default=10000)
    p.add_argument("--mask_ratio", type=float, default=0.15)
    p.add_argument("--eval_cells", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)

    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=6)
    p.add_argument("--MLP_hidden", type=int, default=64)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--dim_feedforward", type=int, default=1024)
    p.add_argument("--cluster_input_dim", type=int, default=4096)
    p.add_argument("--chr_input_dim", type=int, default=25)
    p.add_argument("--modality_input_dim", type=int, default=2)
    p.add_argument("--rank_input_dim", type=int, default=3000)

    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save_json", type=str, default="downstream_test_metrics.json")
    return p.parse_args()


def validate_paths(cfg: Dict):
    required = [
        "base_dir",
        "ENSG2token_path",
        "gene_cluster_info_path",
        "gene_position_info_path",
        "peak2token_path",
        "peak_cluster_path",
        "idf_path",
        "loaded_dict",
    ]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Missing required paths in config: {missing}")


def main():
    args = parse_args()
    cfg = build_config(args)
    validate_paths(cfg)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_ids = load_sample_ids(cfg["loaded_dict"], [x.strip() for x in args.run_ids.split(",") if x.strip()])
    dataset, vocab_size = build_eval_dataset(cfg, sample_ids, rank=0)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = build_model(cfg, vocab_size, device)
    maybe_load_ckpt(model, args.checkpoint, device)

    metrics = evaluate(model, loader, device, cfg["N_gene"], cfg["N_peak"])

    print("\n===== Downstream Test Metrics =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    save_path = Path(args.save_json)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"\n[Saved] {save_path.resolve()}")


if __name__ == "__main__":
    main()
