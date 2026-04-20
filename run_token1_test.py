"""SCOPE-X V3.5.0 with Group Compression - Training Script

Usage:
    # Single GPU (test mode)
    export CUDA_VISIBLE_DEVICES=0
    python run_token1_test.py --experiment_name "test_single" --use_group_compress False --N_gene 100 --N_peak 200 --num_epochs 2

    # Multi-GPU DDP
    python -m torch.distributed.run --nproc_per_node=8 --nnodes=1 run_token1_test.py \\
        --experiment_name "Exp-1_compression_baseline" \\
        --use_group_compress True \\
        --num_groups 64 \\
        --group_mode "merged" \\
        --down_mode "linear" \\
        --N_gene 1000 \\
        --N_peak 2000
"""

import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from model import BERTForSelfSupervised, train_model, plot, plot_individual_result
from utils import MaskedDataset, set_seed, load_multiple_samples, concatenate_samples, load_and_preprocess_test
import pandas as pd
import numpy as np
from datasets import disable_caching


os.environ["NCCL_TIMEOUT"] = "7200"  # Set NCCL timeout to a longer value
current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent


def main():
    parser = argparse.ArgumentParser(description='SCOPE-X Training with Group Compression')

    # Experiment configuration
    parser.add_argument('--experiment_name', type=str, default=None, help='Experiment name for output directory')

    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=256, help='Transformer embedding dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer encoder layers')
    parser.add_argument('--MLP_hidden', type=int, default=64, help='Hidden dimension for MLP decoder')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Dimension of feedforward network')

    # Data configuration
    parser.add_argument('--N_gene', type=int, default=1000, help='Number of genes per cell')
    parser.add_argument('--N_peak', type=int, default=2000, help='Number of peaks per cell')
    parser.add_argument('--gene_position_range', type=int, default=500)
    parser.add_argument('--max_pos_diff', type=int, default=10000)

    # Training configuration
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=60, help='Number of training epochs')
    parser.add_argument('--mask_ratio', type=float, default=0.15, help='Masking ratio for MLM')

    # Loss weights
    parser.add_argument('--loss_weights_value', type=float, default=1.0)
    parser.add_argument('--loss_weights_cluster', type=float, default=1.0)
    parser.add_argument('--loss_weights_chr', type=float, default=0.0)
    parser.add_argument('--loss_weights_modality', type=float, default=0.0)
    parser.add_argument('--loss_weights_atac_bce', type=float, default=1.0)
    parser.add_argument('--alpha_contra', type=float, default=0, help='Contrastive loss weight')

    # Group compression parameters
    parser.add_argument('--use_group_compress', type=lambda x: x.lower() == 'true', default=True,
                        help='Enable group compression in raw feature space')
    parser.add_argument('--group_mode', type=str, default='merged', choices=['merged', 'separate'],
                        help='Group compression mode')
    parser.add_argument('--num_groups', type=int, default=64, help='Number of groups after compression')
    parser.add_argument('--down_mode', type=str, default='linear', choices=['linear', 'gcn', 'gat'],
                        help='Compression method')
    parser.add_argument('--gat_heads', type=int, default=4, help='Number of attention heads for GAT mode')
    parser.add_argument('--perm_seed', type=int, default=42, help='Random seed for permutation')

    # Other
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

    args = parser.parse_args()

    # Set up distributed training
    if args.local_rank == -1:
        # Single GPU mode (for testing)
        args.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
    set_seed(args.local_rank + 1)

    # Configuration dictionary
    config = {
        "gene_position_range": args.gene_position_range,
        "N_gene": args.N_gene,
        "N_peak": args.N_peak,
        "max_pos_diff": args.max_pos_diff,
        "embed_dim": args.embed_dim,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_feedforward": args.dim_feedforward,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "mask_ratio": args.mask_ratio,
        "MLP_hidden": args.MLP_hidden,
        "cluster_input_dim": 4096,
        "chr_input_dim": 25,
        "modality_input_dim": 2,
        "rank_input_dim": 3000,

        # Group compression params
        "use_group_compress": args.use_group_compress,
        "group_mode": args.group_mode,
        "num_groups": args.num_groups,
        "down_mode": args.down_mode,
        "gat_heads": args.gat_heads,
        "perm_seed": args.perm_seed,

        # Loss weights
        "loss_weights": {
            'value': args.loss_weights_value,
            'cluster': args.loss_weights_cluster,
            'chr': args.loss_weights_chr,
            'modality': args.loss_weights_modality,
            'atac_bce': args.loss_weights_atac_bce
        },
        "alpha_contra": args.alpha_contra,

        # Fixed paths (need to be updated for your environment)
        "base_dir": "/XYAIFS00/HDD_POOL/gibh_jkchen/gibh_jkchen_7/guazai/Dataset/scM/processed_data/RNA_ATAC/",
        "ENSG2token_path": "/XYAIFS00/HDD_POOL/gibh_jkchen/gibh_jkchen_7/guazai/SCOPE_X_test/other_file/hm_ENSG2token_dict.pickle",
        "gene_cluster_info_path": "/XYAIFS00/HDD_POOL/gibh_jkchen/gibh_jkchen_7/guazai/SCOPE_X_test/other_file/GRCh38_gene_cluster.csv",
        "gene_position_info_path": "/XYAIFS00/HDD_POOL/gibh_jkchen/gibh_jkchen_7/guazai/SCOPE_X_test/other_file/GRCh38.tss.bed",
        "peak2token_path": "/XYAIFS00/HDD_POOL/gibh_jkchen/gibh_jkchen_7/guazai/SCOPE_X_test/other_file/peak2token_dict.pickle",
        "peak_cluster_path": "/XYAIFS00/HDD_POOL/gibh_jkchen/gibh_jkchen_7/guazai/SCOPE_X_test/other_file/GRCh38_peak_cluster.csv",
        "idf_path": "/XYAIFS00/HDD_POOL/gibh_jkchen/gibh_jkchen_7/guazai/SCOPE_X_test/other_file/peak_IDF_variabilityScore_human.txt",
        "loaded_dict": "/XYAIFS00/HDD_POOL/gibh_jkchen/gibh_jkchen_7/SCOPE-X/SCOPE_X_test/sample_data_select/sample_dict_11w_each_run.json",
        "test_pathtest_base_path": "/XYAIFS00/HDD_POOL/gibh_jkchen/gibh_jkchen_7/guazai/Dataset/evaluation/processed_data/",
        "test_file_name": ['03_hJejunum_10xDemo_10x'],
    }

    # Load sample IDs
    with open(config['loaded_dict'], 'r', encoding='utf-8') as f:
        data = json.load(f)
    sample_ids = [x["sample"] for run_id in ["run_1", "run_2", "run_3"] for x in data[run_id]["samples"]]

    config['sample_num'] = len(sample_ids)
    if args.local_rank == 0:
        print(f"[INFO] Sample count: {len(sample_ids)}", flush=True)
        print(f"[INFO] Group compression: {config['use_group_compress']}", flush=True)
        if config['use_group_compress']:
            print(f"[INFO]   Mode={config['group_mode']}, Groups={config['num_groups']}, DownMode={config['down_mode']}", flush=True)
        print(f"[INFO] N_gene={config['N_gene']}, N_peak={config['N_peak']}", flush=True)

    # Load and preprocess data
    data_dict, *_ , vocab_size = load_multiple_samples(
        sample_ids,
        config['base_dir'],
        config["N_gene"], config["N_peak"],
        config['ENSG2token_path'],
        config['gene_position_info_path'],
        config['gene_cluster_info_path'],
        config["gene_position_range"],
        config['peak2token_path'],
        config['peak_cluster_path'],
        config['idf_path'],
        args.local_rank
    )
    combined_data = concatenate_samples(data_dict)

    if args.local_rank == 0:
        print(f"[INFO] Final input shape: {combined_data.shape}", flush=True)

    # Create dataset and dataloader
    dataset = MaskedDataset(combined_data, config['mask_ratio'], config["N_gene"], config["N_peak"], config["max_pos_diff"])
    sampler = DistributedSampler(dataset, shuffle=True, rank=args.local_rank, num_replicas=dist.get_world_size())
    dataloader = DataLoader(dataset, batch_size=config['batch_size'] // dist.get_world_size(), sampler=sampler)

    config['input_dim'] = combined_data.shape[2]
    config['id_input_dim'] = vocab_size

    if args.local_rank == 0:
        print(f"[INFO] Vocabulary size: {vocab_size}", flush=True)

    # Initialize model with group compression parameters
    model = BERTForSelfSupervised(
        config['input_dim'], config['embed_dim'],
        config['num_layers'], config['MLP_hidden'], config['nhead'],
        config['dim_feedforward'], config['id_input_dim'],
        config['cluster_input_dim'], config['chr_input_dim'], config['modality_input_dim'], config['rank_input_dim'],
        use_group_compress=config.get('use_group_compress', False),
        group_mode=config.get('group_mode', 'merged'),
        num_groups=config.get('num_groups', 32),
        down_mode=config.get('down_mode', 'linear'),
        gat_heads=config.get('gat_heads', 4),
        perm_seed=config.get('perm_seed', 42),
        l_rna=config['N_gene'],
        l_atac=config['N_peak'],
    )

    # Wrap with DDP
    model = DDP(model.to(args.local_rank), device_ids=[args.local_rank], output_device=args.local_rank)

    if args.local_rank == 0:
        print('[INFO] Model initialized', flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Create save directories
    experiment_suffix = f"_{args.experiment_name}" if args.experiment_name else ""
    save_dir = project_root / f"Para_Path_{config['sample_num']}{experiment_suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Train model
    train_loss, train_result = train_model(model, dataloader, optimizer, config, save_dir, args.local_rank)

    config['sample_num'] = len(sample_ids)
    save_dir_result = project_root / f"result{config['sample_num']}{experiment_suffix}"
    os.makedirs(save_dir_result, exist_ok=True)

    if args.local_rank == 0:
        # Save training curves
        np.savetxt(save_dir_result/'loss.csv', train_loss)
        train_result_mat = np.array(train_result).T
        np.savetxt(save_dir_result/'train_result.csv', train_result_mat, delimiter=",",
                   header="precision,recall,f1,atac_acc", comments="")

        plot(train_loss, save_dir_result)
        plot_individual_result(train_result, save_dir_result)

        print(f"[INFO] Results saved to {save_dir_result}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
