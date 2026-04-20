#!/bin/bash
# Exp-2: Extreme Compression (32 groups)
# Purpose: Test more aggressive compression ratio

source /XYAIFS00/gibh_jkchen_7/HOME/miniconda3/bin/activate SCPOEX

python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=1 \
    run_token1_test.py \
    --experiment_name "Exp-2_extreme_compression_32_groups" \
    --use_group_compress True \
    --num_groups 32 \
    --group_mode "merged" \
    --down_mode "linear" \
    --gat_heads 4 \
    --perm_seed 42 \
    --N_gene 1000 \
    --N_peak 2000 \
    --num_epochs 60 \
    --batch_size 64 \
    --embed_dim 256 \
    --num_layers 6 \
    --dim_feedforward 1024 \
    --mask_ratio 0.15 \
    --learning_rate 1e-4 \
    --loss_weights_value 1.0 \
    --loss_weights_cluster 1.0 \
    --loss_weights_chr 0.0 \
    --loss_weights_modality 0.0 \
    --loss_weights_atac_bce 1.0 \
    --alpha_contra 0
