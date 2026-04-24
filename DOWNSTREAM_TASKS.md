# SCOPE-X 下游任务清单与数据字典

本文档给出基于当前 SCOPE-X 双模态框架（RNA + ATAC）的可执行下游任务清单、数据字典与落地脚本映射。

## 1) 任务清单（Checklist）

### A. RNA↔ATAC 跨模态翻译（translation）
- [ ] 数据准备：配对细胞 RNA/ATAC token 序列
- [ ] 训练：RNA→ATAC 与 ATAC→RNA 双向翻译
- [ ] 一致性：可选 cycle consistency（RNA→ATAC→RNA / ATAC→RNA→ATAC）
- [ ] 评估：MSE / PCC / Spearman / AUROC(AUCPR)

### B. 细胞表征学习（representation）
- [ ] 数据准备：cell_type / batch 标签
- [ ] 训练：CLS 监督分类 + 对比损失（可选）
- [ ] 评估：ACC / macro-F1 / NMI / ARI

### C. 跨模态检索（retrieval）
- [ ] 数据准备：RNA-ATAC 成对索引
- [ ] 训练：InfoNCE 或 Triplet 损失
- [ ] 评估：Recall@1/5/10, MRR

### D. Peak-Gene 调控边预测（linkpred）
- [ ] 数据准备：候选 peak-gene 边 + 标签（强监督或弱监督）
- [ ] 训练：二分类 BCE + 距离先验约束（可选）
- [ ] 评估：AUROC / AUPRC

### E. 扰动响应预测（perturb）
- [ ] 数据准备：扰动条件（KO/激活）与响应标签（ΔRNA / ΔATAC）
- [ ] 训练：回归或分类损失
- [ ] 评估：PCC / Spearman / Top-k overlap

---

## 2) 统一样本数据字典（JSON/Parquet/Arrow 均可）

每条样本建议字段：

| 字段名 | 类型 | 形状/示例 | 说明 |
|---|---|---|---|
| `cell_id` | str | `"cell_0001"` | 细胞唯一ID |
| `sample_id` | str | `"GRCh38.M00096"` | 样本ID |
| `rna_tokens` | int/float list | `[N_gene, 6]` | RNA token序列，列顺序见下 |
| `atac_tokens` | int/float list | `[N_peak, 6]` | ATAC token序列，列顺序见下 |
| `pair_id` | str/int | `"pair_0001"` | RNA-ATAC 配对ID（同细胞同pair） |
| `cell_type` | str/int | `"B_cell"` | 细胞类型标签（可选） |
| `batch` | str/int | `"run_1"` | 批次标签（可选） |
| `split` | str | `"train"/"val"/"test"` | 数据分割 |
| `candidate_edges` | list | `[[peak_id,gene_id,dist], ...]` | linkpred候选边（可选） |
| `edge_labels` | list[int] | `[0,1,0...]` | linkpred监督标签（可选） |
| `perturb` | dict | `{target:"GATA1",type:"KO"}` | 扰动条件（可选） |
| `delta_rna` | list[float] | `[N_gene]` | 扰动RNA响应标签（可选） |
| `delta_atac` | list[float] | `[N_peak]` | 扰动ATAC响应标签（可选） |

### token 列定义（与现有仓库一致）
`[id, value, chr, cluster, modality, rank]`

> 注：当前训练中 `modality` 由 `utils.MaskedDataset` 动态拼接，可在下游数据准备阶段显式存储，简化推理管线。

---

## 3) 任务特定字典

### 3.1 translation

| 字段 | 类型 | 必需 |
|---|---|---|
| `rna_tokens` | `[N_gene,6]` | 是 |
| `atac_tokens` | `[N_peak,6]` | 是 |
| `pair_id` | str/int | 是 |
| `split` | str | 是 |

### 3.2 representation

| 字段 | 类型 | 必需 |
|---|---|---|
| `rna_tokens` | `[N_gene,6]` | 是 |
| `atac_tokens` | `[N_peak,6]` | 是 |
| `cell_type` | int | 是 |
| `batch` | int/str | 否 |
| `split` | str | 是 |

### 3.3 retrieval

| 字段 | 类型 | 必需 |
|---|---|---|
| `rna_tokens` | `[N_gene,6]` | 是 |
| `atac_tokens` | `[N_peak,6]` | 是 |
| `pair_id` | str/int | 是 |
| `split` | str | 是 |

### 3.4 linkpred

| 字段 | 类型 | 必需 |
|---|---|---|
| `rna_tokens` | `[N_gene,6]` | 是 |
| `atac_tokens` | `[N_peak,6]` | 是 |
| `candidate_edges` | `[[peak_id,gene_id,distance], ...]` | 是 |
| `edge_labels` | `[0/1,...]` | 是 |
| `split` | str | 是 |

### 3.5 perturb

| 字段 | 类型 | 必需 |
|---|---|---|
| `rna_tokens` | `[N_gene,6]` | 是 |
| `atac_tokens` | `[N_peak,6]` | 是 |
| `perturb` | dict | 是 |
| `delta_rna` | `[N_gene]` | 任务定义相关 |
| `delta_atac` | `[N_peak]` | 任务定义相关 |
| `split` | str | 是 |

---

## 4) 脚本映射

- `run_downstream.py`：统一下游训练入口
- `downstream_dataset.py`：下游数据集与collate
- `downstream_heads.py`：任务头
- `downstream_losses.py`：任务损失

