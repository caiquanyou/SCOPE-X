"""Unified downstream training entry for SCOPE-X.

Example:
python run_downstream.py \
  --task representation \
  --manifest ./data/downstream_manifest.json \
  --split train \
  --epochs 5 \
  --batch_size 8 \
  --num_classes 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import BERTForSelfSupervised
from downstream_dataset import (
    load_manifest,
    TranslationDataset,
    RepresentationDataset,
    RetrievalDataset,
    LinkPredictionDataset,
    PerturbationDataset,
    collate_tokens,
)
from downstream_heads import (
    CrossModalTranslatorHead,
    RepresentationClassifierHead,
    RetrievalProjectionHead,
    PeakGeneLinkHead,
    PerturbationResponseHead,
)
from downstream_losses import (
    translation_loss,
    cycle_consistency_loss,
    retrieval_infonce_loss,
    representation_cls_loss,
    link_prediction_loss,
    perturbation_regression_loss,
    weighted_sum,
)


def build_model(device: torch.device, l_rna: int, l_atac: int, embed_dim: int = 256) -> BERTForSelfSupervised:
    model = BERTForSelfSupervised(
        input_dim=6,
        embed_dim=embed_dim,
        num_layers=4,
        MLP_hidden=64,
        nhead=8,
        dim_feedforward=1024,
        id_input_dim=400000,
        cluster_input_dim=4096,
        chr_input_dim=25,
        modality_input_dim=2,
        rank_input_dim=3000,
        use_group_compress=False,
        l_rna=l_rna,
        l_atac=l_atac,
    )
    return model.to(device)


def canonicalize_modality(rna: torch.Tensor, atac: torch.Tensor) -> torch.Tensor:
    """Return concatenated tokens in model format [id, value, chr, cluster, modality, rank]."""
    rna = rna.clone()
    atac = atac.clone()

    # If column 5 exists, treat it as rank. Always rewrite modality column.
    if rna.shape[-1] < 6 or atac.shape[-1] < 6:
        raise ValueError("Expected token dim >= 6.")

    rna[:, 4] = 0
    atac[:, 4] = 1
    return torch.cat([rna[:, :6], atac[:, :6]], dim=0)


def encode_batch(backbone: BERTForSelfSupervised, batch: Dict[str, torch.Tensor], device: torch.device):
    rna = batch["rna_tokens"].to(device)
    atac = batch["atac_tokens"].to(device)

    merged = []
    for i in range(rna.shape[0]):
        merged.append(canonicalize_modality(rna[i], atac[i]))
    x = torch.stack(merged, dim=0)

    logits, proj_rna, proj_atac, cls_rna, cls_atac = backbone(x)
    return logits, proj_rna, proj_atac, cls_rna, cls_atac


def train_one_epoch(task, model, heads, loader, optimizer, device, cfg):
    model.train()
    for m in heads.values():
        m.train()

    total = 0.0
    steps = 0
    for batch in loader:
        optimizer.zero_grad()
        mlm_logits, proj_rna, proj_atac, cls_rna, cls_atac = encode_batch(model, batch, device)

        losses = {}

        if task == "representation":
            logits = heads["rep_head"](cls_rna, cls_atac)
            labels = batch["cell_type"].to(device)
            losses["cls"] = representation_cls_loss(logits, labels)
            losses["retrieval_aux"] = retrieval_infonce_loss(proj_rna, proj_atac) * cfg.aux_infonce

        elif task == "retrieval":
            z_rna, z_atac = heads["ret_head"](cls_rna, cls_atac)
            losses["infonce"] = retrieval_infonce_loss(z_rna, z_atac, cfg.temperature)

        elif task == "translation":
            # RNA->ATAC value prediction using ATAC token embeddings from decoder output
            atac_pred = heads["trans_a"](mlm_logits["value"])  # [B, L, 1] -> [B, L, 1]
            target = mlm_logits["value"].detach()
            losses["r2a"] = translation_loss(atac_pred, target, is_binary=False)

            rna_pred = heads["trans_r"](mlm_logits["value"])
            losses["a2r"] = translation_loss(rna_pred, target, is_binary=False)
            losses["cycle"] = cycle_consistency_loss(rna_pred, atac_pred) * cfg.cycle_weight

        elif task == "linkpred":
            edges = batch["candidate_edges"].to(device)  # [B, E, 3]
            labels = batch["edge_labels"].to(device)     # [B, E]
            bsz, edge_n, _ = edges.shape

            # Simple: use CLS as context and edge distance as feature.
            peak_embed = cls_atac.unsqueeze(1).expand(bsz, edge_n, -1).reshape(-1, cls_atac.shape[-1])
            gene_embed = cls_rna.unsqueeze(1).expand(bsz, edge_n, -1).reshape(-1, cls_rna.shape[-1])
            distance = edges[:, :, 2].reshape(-1, 1)

            logits = heads["link_head"](peak_embed, gene_embed, distance).view(bsz, edge_n)
            losses["bce"] = link_prediction_loss(logits, labels)

        elif task == "perturb":
            if "delta_rna" in batch:
                pred = heads["pert_head"](cls_rna, cls_atac)
                target = batch["delta_rna"].to(device)
                losses["delta"] = perturbation_regression_loss(pred, target)
            elif "delta_atac" in batch:
                pred = heads["pert_head"](cls_rna, cls_atac)
                target = batch["delta_atac"].to(device)
                losses["delta"] = perturbation_regression_loss(pred, target)
            else:
                raise ValueError("Perturb task requires delta_rna or delta_atac in batch.")

        else:
            raise ValueError(f"Unsupported task: {task}")

        loss = weighted_sum(losses)
        loss.backward()
        optimizer.step()

        total += loss.item()
        steps += 1

    return total / max(steps, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["translation", "representation", "retrieval", "linkpred", "perturb"])
    parser.add_argument("--manifest", type=str, required=True, help="Path to JSON list of samples")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--cycle_weight", type=float, default=0.2)
    parser.add_argument("--aux_infonce", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="./downstream_outputs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = load_manifest(args.manifest)

    task2dataset = {
        "translation": TranslationDataset,
        "representation": RepresentationDataset,
        "retrieval": RetrievalDataset,
        "linkpred": LinkPredictionDataset,
        "perturb": PerturbationDataset,
    }
    dataset = task2dataset[args.task](samples, split=args.split)
    if len(dataset) == 0:
        raise ValueError(f"No samples found for split={args.split}.")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tokens)

    first = dataset[0]
    l_rna = first["rna_tokens"].shape[0]
    l_atac = first["atac_tokens"].shape[0]

    model = build_model(device, l_rna=l_rna, l_atac=l_atac, embed_dim=args.embed_dim)
    heads: Dict[str, nn.Module] = {}

    if args.task == "representation":
        heads["rep_head"] = RepresentationClassifierHead(args.embed_dim, args.num_classes).to(device)
    elif args.task == "retrieval":
        heads["ret_head"] = RetrievalProjectionHead(args.embed_dim, proj_dim=128).to(device)
    elif args.task == "translation":
        heads["trans_a"] = CrossModalTranslatorHead(1, hidden_dim=16, out_dim=1).to(device)
        heads["trans_r"] = CrossModalTranslatorHead(1, hidden_dim=16, out_dim=1).to(device)
    elif args.task == "linkpred":
        heads["link_head"] = PeakGeneLinkHead(args.embed_dim).to(device)
    elif args.task == "perturb":
        out_dim = first.get("delta_rna", first.get("delta_atac")).shape[0]
        heads["pert_head"] = PerturbationResponseHead(args.embed_dim, out_dim=out_dim).to(device)

    params = list(model.parameters())
    for h in heads.values():
        params.extend(list(h.parameters()))

    optimizer = torch.optim.AdamW(params, lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(args.task, model, heads, loader, optimizer, device, args)
        log = {"epoch": epoch, "loss": loss, "task": args.task}
        history.append(log)
        print(json.dumps(log, ensure_ascii=False), flush=True)

    ckpt = {
        "backbone": model.state_dict(),
        "heads": {k: v.state_dict() for k, v in heads.items()},
        "args": vars(args),
        "history": history,
    }
    torch.save(ckpt, out_dir / f"{args.task}_checkpoint.pt")
    with open(out_dir / f"{args.task}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
