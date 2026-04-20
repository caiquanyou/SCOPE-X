# SCOPE-X V3.5.0 with Group Compression - Quick Start Guide

## 📋 Overview

This codebase merges the **RandomGroupCodec** sequence compression from `grouping/` into the **SCOPE-X V3.5.0** production architecture with the following key improvements:

### Key Features
- ✅ **Dual CLS tokens** (RNA_CLS, ATAC_CLS) for contrastive learning
- ✅ **Multi-task decoder** (value, cluster, chr, modality prediction)
- ✅ **Optional group compression in RAW feature space** `[B, L, 4096] → [B, gc, 4096]`
- ✅ **Fixed peak status prediction**: Uses ALL ATAC tokens (not just masked ones)
- ✅ **DDP multi-GPU training support**

### Critical Fixes Applied

#### 1. Peak Status Prediction Loss (Line 276 in V3.5.0)
**Before** (❌ Wrong):
```python
is_atac_mask = loss_mask & (target_dict['modality'] == 1)  # Only masked tokens
```

**After** (✅ Correct):
```python
is_atac = (target_dict['modality'] == 1)  # Use ALL ATAC tokens
```

**Why**: Peak status prediction is a discriminative task independent of MLM masking. Using only masked tokens wastes ~85% of training signal and causes "all closed" or "all open" trivial solutions.

#### 2. Compression Position
Compression happens in **RAW feature space** BEFORE embedding:
```
Input: [B, L=3000, input_dim=4096]
  ↓ compress (RandomGroupCodec)
[B, gc=64, 4096]
  ↓ embedding
[B, gc=64, embed_dim=512]
  ↓ transformer
[B, gc+2=66, embed_dim=512]
  ↓ expand_embed
[B, L=3000, embed_dim=512]
  ↓ MultiDecoder
Output: predictions
```

## 🚀 Quick Start

### Prerequisites

1. **Copy missing dependencies** (if not already present):
   ```bash
   cd E:\caiqy\多组学大模型\SCOPE-X\grouping\20260403_SCOPE-X_V3.5.0_with_grouping_fixed
   
   # Check if these files exist
   ls ATACtoken.py RNAtoken.py anlysis.py utils.py run_code.sh
   ```
   
   If any are missing, copy them from the V3.5.0 directory:
   ```bash
   cp ../20260403_SCOPE-X_V3.5.0_parameter_test2/20260403_SCOPE-X_V3.5.0_parameter_test2/ATACtoken.py .
   cp ../20260403_SCOPE-X_V3.5.0_parameter_test2/20260403_SCOPE-X_V3.5.0_parameter_test2/RNAtoken.py .
   cp ../20260403_SCOPE-X_V3.5.0_parameter_test2/20260403_SCOPE-X_V3.5.0_parameter_test2/anlysis.py .
   ```

2. **Update data paths** in `run_token1_test.py`:
   Edit the `config` dictionary to point to your actual data locations:
   ```python
   config = {
       "base_dir": "/your/actual/data/path",
       "ENSG2token_path": "/your/actual/path/hm_ENSG2token_dict.pickle",
       # ... other paths
   }
   ```

3. **Verify conda environment**:
   The scripts use `/XYAIFS00/gibh_jkchen_7/HOME/miniconda3/bin/activate SCPOEX`.
   Update this path in each script if needed.

### Single GPU Test (Recommended First)

Test on a single GPU to verify everything works:

```bash
export CUDA_VISIBLE_DEVICES=0
cd E:\caiqy\多组学大模型\SCOPE-X\grouping\20260403_SCOPE-X_V3.5.0_with_grouping_fixed

python run_token1_test.py \
    --experiment_name "test_single_gpu" \
    --use_group_compress False \
    --N_gene 100 \
    --N_peak 200 \
    --num_epochs 2 \
    --batch_size 64 \
    --embed_dim 256 \
    --mask_ratio 0.15
```

Expected output:
- Model initializes successfully
- Training starts without errors
- Loss decreases over 2 epochs
- Results saved to `result*_test_single_gpu/`

### Multi-GPU DDP Training

Once single GPU test passes, run full experiments:

```bash
# Example: Exp-1 (Compression Baseline)
cd E:\caiqy\多组学大模型\SCOPE-X\grouping\20260403_SCOPE-X_V3.5.0_with_grouping_fixed
bash scripts/run_Exp-1.sh
```

## 🧪 Experimental Scripts

All 13 experiment scripts are in `scripts/` directory:

| Script | Purpose | Key Parameters |
|--------|---------|----------------|
| `run_Exp-0.sh` | **Baseline (No Compression)** | `--use_group_compress False` |
| `run_Exp-1.sh` | **Compression Baseline** | 64 groups, linear mode |
| `run_Exp-2.sh` | Extreme Compression | 32 groups |
| `run_Exp-3.sh` | Light Compression | 128 groups |
| `run_Exp-4.sh` | Minimal Compression | 256 groups |
| `run_Exp-5.sh` | Separate Mode | RNA/ATAC compressed independently |
| `run_Exp-6.sh` | GCN Aggregation | GCN-based grouping |
| `run_Exp-7.sh` | GAT Attention | GAT-based grouping with 4 heads |
| `run_Exp-8.sh` | Seed 123 | Different permutation seed |
| `run_Exp-9.sh` | Seed 999 | Another permutation seed |
| `run_Exp-B.sh` | Token Doubling | N_gene=2000, N_peak=4000 |
| `run_Exp-C.sh` | Token Extreme | N_gene=3000, N_peak=6000 |
| `run_Exp-D.sh` | Token Halving | N_gene=500, N_peak=1000 |

## 🔧 Custom Experiments

You can customize experiments via command line arguments:

```bash
python run_token1_test.py \
    --experiment_name "my_custom_exp" \
    --use_group_compress True \
    --num_groups 64 \
    --group_mode "merged" \
    --down_mode "linear" \
    --gat_heads 4 \
    --perm_seed 42 \
    --N_gene 1000 \
    --N_peak 2000 \
    --num_epochs 60 \
    --batch_size 64 \
    --embed_dim 256 \
    --mask_ratio 0.15 \
    --learning_rate 1e-4 \
    --loss_weights_value 1.0 \
    --loss_weights_cluster 1.0 \
    --loss_weights_chr 0.0 \
    --loss_weights_modality 0.0 \
    --loss_weights_atac_bce 1.0 \
    --alpha_contra 0
```

### Available Arguments

#### Model Architecture
- `--embed_dim`: Transformer embedding dimension (default: 256)
- `--num_layers`: Number of transformer layers (default: 6)
- `--MLP_hidden`: Hidden dimension for decoder MLP (default: 64)
- `--nhead`: Number of attention heads (default: 8)
- `--dim_feedforward`: Feedforward network dimension (default: 1024)

#### Data Configuration
- `--N_gene`: Number of genes per cell (default: 1000)
- `--N_peak`: Number of peaks per cell (default: 2000)
- `--mask_ratio`: Masking ratio for MLM (default: 0.15)

#### Group Compression
- `--use_group_compress`: Enable compression (True/False, default: True)
- `--group_mode`: Compression mode ("merged" or "separate", default: merged)
- `--num_groups`: Number of groups after compression (default: 64)
- `--down_mode`: Compression method ("linear", "gcn", or "gat", default: linear)
- `--gat_heads`: Number of attention heads for GAT mode (default: 4)
- `--perm_seed`: Random seed for permutation (default: 42)

#### Training
- `--batch_size`: Batch size per GPU (default: 64)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--num_epochs`: Number of epochs (default: 60)

#### Loss Weights
- `--loss_weights_value`: Value prediction weight (default: 1.0)
- `--loss_weights_cluster`: Cluster prediction weight (default: 1.0)
- `--loss_weights_chr`: Chromosome prediction weight (default: 0.0)
- `--loss_weights_modality`: Modality prediction weight (default: 0.0)
- `--loss_weights_atac_bce`: ATAC BCE loss weight (default: 1.0)
- `--alpha_contra`: Contrastive loss weight (default: 0)

## 📊 Output Structure

After running an experiment, results are saved to:

```
Para_Path_<sample_num>_<experiment_name>/
├── <sample>_sample_model_epoch1_lossX.XXXX.pth
├── <sample>_sample_model_epoch2_lossX.XXXX.pth
└── ...

result<sample_num>_<experiment_name>/
├── loss.csv              # Per-epoch loss values
├── train_result.csv      # P/R/F1/Acc per epoch
├── loss_curve.png        # Training loss plot
├── result_precision.png  # Precision curve
├── result_recall.png     # Recall curve
├── result_f1.png         # F1 curve
└── result_avg_acc.png    # Accuracy curve
```

## 🐛 Troubleshooting

### Issue: Import Error for ATACtoken/RNAtoken

**Solution**: Copy missing files from V3.5.0 directory (see Prerequisites above).

### Issue: Path Not Found

**Solution**: Update data paths in `run_token1_test.py` config dictionary to match your environment.

### Issue: DDP Initialization Hangs

**Solutions**:
1. Check NCCL configuration: `export NCCL_DEBUG=INFO`
2. Verify GPU availability: `nvidia-smi`
3. Ensure port 29500 is not occupied
4. Try reducing number of GPUs: `--nproc_per_node=4`

### Issue: Out of Memory

**Solutions**:
1. Reduce batch size: `--batch_size 32`
2. Reduce token length: `--N_gene 500 --N_peak 1000`
3. Reduce model size: `--embed_dim 128 --dim_feedforward 512`
4. Reduce number of groups: `--num_groups 32`

### Issue: Peak Prediction Always "Closed" or "Open"

**This has been fixed!** The fix uses ALL ATAC tokens instead of just masked ones. See "Critical Fixes Applied" section above.

## 📝 Code Architecture

### File Structure

```
20260403_SCOPE-X_V3.5.0_with_grouping_fixed/
├── model.py                 # Core model with group compression
├── utils.py                 # Data loading and utilities
├── run_token1_test.py       # Main training script
├── anlysis.py               # Analysis utilities
├── ATACtoken.py             # ATAC preprocessing (dependency)
├── RNAtoken.py              # RNA preprocessing (dependency)
├── run_code.sh              # Legacy runner
├── scripts/
│   ├── run_Exp-0.sh         # Baseline (no compression)
│   ├── run_Exp-1.sh         # Compression baseline
│   ├── ...                  # Other experiments
│   └── run_Exp-D.sh         # Token halving
└── README.md                # This file
```

### Key Classes

#### RandomGroupCodec (model.py:168-328)
Handles sequence compression/expansion:
- `compress(x)`: [B, L, D] → [B, gc, D]
- `expand_embed(z)`: [B, gc, E] → [B, L, E]

#### BERTForSelfSupervised (model.py:386-537)
Main model architecture:
- Dual CLS tokens for contrastive learning
- Optional group compression before embedding
- Multi-task decoder for MLM + peak status prediction

#### train_model (model.py:539-704)
Training loop with fixed peak prediction:
- Uses ALL ATAC tokens for BCE loss (line 608)
- Supports DDP distributed training
- Logs detailed metrics per epoch

##  Recommended Experiment Order

1. **Start with Exp-0** (baseline without compression) to verify setup
2. **Run Exp-1** (compression baseline) to validate compression works
3. **Test different compression ratios**: Exp-2, Exp-3, Exp-4
4. **Try different modes**: Exp-5 (separate), Exp-6 (GCN), Exp-7 (GAT)
5. **Test robustness**: Exp-8, Exp-9 (different seeds)
6. **Scale up**: Exp-B, Exp-C (longer sequences), Exp-D (shorter)

## 📞 Support

If you encounter issues not covered in this guide:

1. Check error messages carefully
2. Verify all prerequisites are met
3. Run single GPU test first
4. Provide:
   - Complete error message
   - Command used
   - Environment info (GPU count, conda path, etc.)

---

**Last Updated**: April 14, 2026
**Version**: V3.5.0 with Group Compression (Fixed)
**Status**: ✅ Ready for experimentation
