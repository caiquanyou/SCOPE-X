from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import torch
from torch.utils.data import Dataset


@dataclass
class DownstreamSample:
    cell_id: str
    sample_id: str
    rna_tokens: List[List[float]]
    atac_tokens: List[List[float]]
    pair_id: Optional[str] = None
    cell_type: Optional[int] = None
    batch: Optional[str] = None
    split: str = "train"
    candidate_edges: Optional[List[List[float]]] = None
    edge_labels: Optional[List[int]] = None
    perturb: Optional[Dict[str, Any]] = None
    delta_rna: Optional[List[float]] = None
    delta_atac: Optional[List[float]] = None


def load_manifest(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list of sample dictionaries.")
    return data


class BaseDownstreamDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], split: str = "train"):
        self.samples = [s for s in samples if s.get("split", "train") == split]

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _to_tensor(x, dtype=torch.float32):
        return torch.tensor(x, dtype=dtype)


class TranslationDataset(BaseDownstreamDataset):
    def __getitem__(self, idx):
        s = self.samples[idx]
        rna = self._to_tensor(s["rna_tokens"])
        atac = self._to_tensor(s["atac_tokens"])
        return {
            "rna_tokens": rna,
            "atac_tokens": atac,
            "pair_id": s.get("pair_id", ""),
            "cell_id": s.get("cell_id", str(idx)),
        }


class RepresentationDataset(BaseDownstreamDataset):
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "rna_tokens": self._to_tensor(s["rna_tokens"]),
            "atac_tokens": self._to_tensor(s["atac_tokens"]),
            "cell_type": torch.tensor(s["cell_type"], dtype=torch.long),
            "cell_id": s.get("cell_id", str(idx)),
        }


class RetrievalDataset(BaseDownstreamDataset):
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "rna_tokens": self._to_tensor(s["rna_tokens"]),
            "atac_tokens": self._to_tensor(s["atac_tokens"]),
            "pair_id": s.get("pair_id", str(idx)),
        }


class LinkPredictionDataset(BaseDownstreamDataset):
    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "rna_tokens": self._to_tensor(s["rna_tokens"]),
            "atac_tokens": self._to_tensor(s["atac_tokens"]),
            "candidate_edges": self._to_tensor(s["candidate_edges"]),
            "edge_labels": torch.tensor(s["edge_labels"], dtype=torch.float32),
            "cell_id": s.get("cell_id", str(idx)),
        }


class PerturbationDataset(BaseDownstreamDataset):
    def __getitem__(self, idx):
        s = self.samples[idx]
        out = {
            "rna_tokens": self._to_tensor(s["rna_tokens"]),
            "atac_tokens": self._to_tensor(s["atac_tokens"]),
            "cell_id": s.get("cell_id", str(idx)),
            "perturb": s.get("perturb", {}),
        }
        if s.get("delta_rna") is not None:
            out["delta_rna"] = self._to_tensor(s["delta_rna"])
        if s.get("delta_atac") is not None:
            out["delta_atac"] = self._to_tensor(s["delta_atac"])
        return out


def pad_stack_3d(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    b = len(tensors)
    max_len = max(t.shape[0] for t in tensors)
    feat = tensors[0].shape[1]
    out = torch.full((b, max_len, feat), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.shape[0], :] = t
    return out


def collate_tokens(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [x[k] for x in batch]
        if isinstance(vals[0], torch.Tensor):
            if vals[0].dim() == 2:
                out[k] = pad_stack_3d(vals)
            else:
                out[k] = torch.stack(vals)
        else:
            out[k] = vals
    return out
