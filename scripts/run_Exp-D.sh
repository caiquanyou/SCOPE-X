#!/bin/bash
# Exp-D: Token Halving (500 genes, 1000 peaks)
# Purpose: Test shorter token sequences with compression

source /XYAIFS00/gibh_jkchen_7/HOME/miniconda3/bin/activate SCPOEX

python -m torch.distributed.run \
    --nproc_per_node=8 \
    --nnodes=1 \
    run_token1_test.py \
    --experiment_name "Exp-D_token_halving" \
    --use_group_compress True \
    --num_groups 64 \
    --group_mode "merged" \
    --down_mode "linear" \
    --gat_heads 4 \
    --perm_seed 42 \
    --N_gene 500 \
    --N_peak 1000 \
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
