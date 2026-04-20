import scanpy as sc
import torch
import numpy as np
from torch.utils.data import Dataset
import os
import random
import sys
from datasets import load_from_disk
from ATACtoken import PeakProcessor
from RNAtoken import GeneProcessor
from anlysis import cosine_similarity,graph_connectivity,calculate_metrics
import matplotlib.pyplot as plt
import torch.distributed as dist
from typing import List
import time
import pandas as pd
def set_seed(seed):
    # Python内置随机模块
    random.seed(seed)
    
    # NumPy随机模块
    np.random.seed(seed)
    
    # PyTorch随机模块
    torch.manual_seed(seed)
    
    # CUDA（如果使用 GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  #多 GPU时设置所有 GPU的种子
        
        #确保 cuDNN的行为一致（可能会影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Logger:
   def __init__(self, filename="output.log"):
      self.terminal = sys.stdout  # 保存原始 stdout
      self.log = open(filename, "w", encoding="utf-8")

   def write(self, message):
      self.terminal.write(message)  # 输出到终端
      self.log.write(message)       # 写入文件

   def flush(self):
      pass  # 避免文件关闭时的错误


# Load and preprocess data
def load_and_preprocess(dataset_path,N_gene,N_peak,ENSG2token_path,gene_position_info_path,gene_cluster_info_path,gene_position_range,peak2token_path,peak_cluster_path,idf_path):
    # Load RNA data
    #datset_rna_atac= load_from_disk(dataset_path)
    #rna_tokens,gene_positions=rna_token_build_parallel(datset_rna_atac,N_gene,ENSG2token_path,gene_position_info_path,gene_cluster_info_path,gene_position_range)
    rna_processor = GeneProcessor(dataset_path,ENSG2token_path,gene_cluster_info_path,gene_position_info_path)
    rna_dataset = rna_processor.process(K=N_gene)
    rna_tokens = rna_processor.get_topk_array()
    rna_positions = rna_dataset['gene_position']
    
    atac_processor = PeakProcessor(dataset_path,peak2token_path,ENSG2token_path,peak_cluster_path,idf_path)
    atac_dataset, vocab_size = atac_processor.process(K=N_peak)
    atac_tokens = atac_processor.get_topk_array()
    atac_positions = atac_dataset['atac_position']
    
    
    #vocab_size=183343
    
   

    # Concatenate RNA and ATAC data
    combined_data = np.concatenate((rna_tokens, atac_tokens),axis=1)
    return combined_data,rna_positions,atac_positions,vocab_size

#def masked_data(data,masked_ratio):

class MaskedDataset(Dataset):
    def __init__(self, input_data,masked_ratio,N_gene,N_peak,max_pos_diff):
        self.input_data = input_data
        self.target_data = input_data
        self.masked_ratio=masked_ratio
        self.max_pos_diff = max_pos_diff
        self.N_gene = N_gene
        self.N_peak = N_peak

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]

        modality_ids = [0] * self.N_gene + [1] * self.N_peak
        modality_ids = np.array(modality_ids).reshape(-1, 1)
        input_sample = np.concatenate((input_sample, modality_ids), axis=1)
        target_sample = np.concatenate((target_sample[:,:4], modality_ids, target_sample[:, 5:6]), axis=1)

        '''
        mask = np.random.rand(len(input_sample)) < self.masked_ratio 
        mask = mask.reshape(-1, 1)
        mask = np.tile(mask, (1, input_sample.shape[1]))

        masked_sample = input_sample.copy()
        masked_sample[mask] = 0
        loss_mask = mask.astype(bool)

        '''
        rna = input_sample[:self.N_gene]
        atac = input_sample[self.N_gene:]

        masked_rna, masked_atac = hybrid_masking_pair_then_random(rna, atac, self.masked_ratio, self.max_pos_diff)


        full_mask = np.ones((self.N_gene + self.N_peak, 1), dtype=float) # [K, 1]，初始化全是 1
        full_mask[masked_rna] = 0.0 # mask 的 token 行：mask 值 = 0 不 mask 的 token 行：mask 值 = 1
        full_mask[[self.N_gene + i for i in masked_atac]] = 0.0

        #不给modality进行mask
        masked_sample = target_sample * full_mask # [K, 6]
        masked_sample[:, 4] = target_sample[:, 4]#还原模态
        masked_sample[:, 2] = target_sample[:, 2]#还原染色体
        #masked_sample=np.concatenate((mask,target_sample[:,4:]), axis=1)
        
        loss_mask = np.zeros((self.N_gene + self.N_peak,), dtype=bool) # loss_mask 形状是 (L,)（一维）
        loss_mask[masked_rna] = True # 标记被掩码的RNA token
        loss_mask[[self.N_gene + i for i in masked_atac]] = True # 标记被掩码的ATAC token，最终loss_mask = [False, True, False, True, False...]


        #return torch.tensor(masked_sample), torch.tensor(target_sample), torch.tensor(loss_mask, dtype=torch.bool)
        return torch.from_numpy(masked_sample).float(), torch.from_numpy(target_sample).float(), torch.from_numpy(loss_mask).bool() #这省内存，会更慢
    
# 缓存文件前缀
CACHE_FILE_PREFIX = "cache-"
def get_all_cache_files(data_path) -> List[str]:
    cache_files = [
        os.path.join(data_path, fname)
        for fname in os.listdir(data_path)
        if fname.startswith(CACHE_FILE_PREFIX) and fname.endswith(".arrow")
    ]
    return sorted(cache_files)  # 排序保证各rank读取顺序一致

def check_cache_exists(data_path) -> bool:
    """判断缓存是否存在（至少有1个 .arrow 缓存文件）"""
    cache_files = get_all_cache_files(data_path)
    return len(cache_files) > 0

def load_multiple_samples(sample_ids,base_dir,N_gene,N_peak,ENSG2token_path,gene_position_info_path,gene_cluster_info_path,gene_position_range,peak2token_path,peak_cluster_path,idf_path,rank):
    """加载多个样本数据，返回字典 {sample_id: numpy_array}"""
    data_dict = {}
    gene_position_id={}
    atac_position_id={}
    vocab_size = 0

    for sample_id in sample_ids:
        # 动态生成路径
        data_path = os.path.join(base_dir,f"GRCh38.{sample_id}",'RNA_ATAC_data_v1')
        
        if os.path.exists(data_path):
            print(f"Rank {rank} Processing sample: {sample_id}")
            try:
                # 处理单个样本
                #先缓存
                '''
                if rank == 0:
                   sample_matrix,gene_positions,atac_positions, vocab_size = load_and_preprocess(data_path,N_gene,N_peak,ENSG2token_path,gene_position_info_path,gene_cluster_info_path,gene_position_range,peak2token_path,peak_cluster_path,idf_path)
                   time.sleep(30)
                dist.barrier()
                if rank!=0:
                   
                   sample_matrix,gene_positions,atac_positions, vocab_size = load_and_preprocess(data_path,N_gene,N_peak,ENSG2token_path,gene_position_info_path,gene_cluster_info_path,gene_position_range,peak2token_path,peak_cluster_path,idf_path)
                '''
                sample_matrix,gene_positions,atac_positions, vocab_size = load_and_preprocess(data_path,N_gene,N_peak,ENSG2token_path,gene_position_info_path,gene_cluster_info_path,gene_position_range,peak2token_path,peak_cluster_path,idf_path)
                data_dict[sample_id] = sample_matrix
                #gene_position_id[sample_id] = gene_positions
                #atac_position_id[sample_id] = atac_positions
            except Exception as e:
                print(f"Error processing {sample_id}: {str(e)}")
        else:
            print(f"Missing files for sample: {sample_id}")
    
    return data_dict,gene_position_id,atac_position_id, vocab_size
    


# 定义数据合并函数
def concatenate_samples(data_dict):
    """纵向拼接所有样本数据（按细胞维度）"""
    sample_matrices = list(data_dict.values())
    
    # 检查特征维度是否一致
    n_features = sample_matrices[0].shape[1]
    for mat in sample_matrices[1:]:
        assert mat.shape[1] == n_features, "特征维度不一致！"
    
    # 纵向拼接（假设所有样本的特征维度相同）
    combined_matrix = np.concatenate(sample_matrices, axis=0)
    return combined_matrix


def load_and_preprocess_test(dataset_path, N_gene,N_peak,ENSG2token_path,gene_position_info_path,gene_cluster_info_path,gene_position_range,peak2token_path,peak_cluster_path,idf_path):
    # Load RNA data
    datset_rna_atac= load_from_disk(dataset_path)
    test_cell_type=datset_rna_atac['cell_types']
    
    rna_processor = GeneProcessor(dataset_path,ENSG2token_path,gene_cluster_info_path,gene_position_info_path)
    rna_dataset = rna_processor.process(K=N_gene)
    rna_tokens = rna_processor.get_topk_array()
    rna_positions = rna_dataset['gene_position']
    
    atac_processor = PeakProcessor(dataset_path,peak2token_path,ENSG2token_path,peak_cluster_path,idf_path)
    atac_dataset, vocab_size = atac_processor.process(K=N_peak)
    atac_tokens = atac_processor.get_topk_array()
    atac_positions = atac_dataset['atac_position']

    # Concatenate RNA and ATAC data
    combined_data = np.concatenate((rna_tokens, atac_tokens),axis=1)
    return combined_data,rna_positions,atac_positions,vocab_size,test_cell_type

def load_and_preprocess_test_peak(dataset_path, N_gene,N_peak,ENSG2token_path,gene_position_info_path,gene_cluster_info_path,gene_position_range,peak2token_path,peak_cluster_path,idf_path):
    # Load RNA data
    datset_rna_atac= load_from_disk(dataset_path)
    test_cell_type=datset_rna_atac['cell_types']
    
    rna_processor = GeneProcessor(dataset_path,ENSG2token_path,gene_cluster_info_path,gene_position_info_path)
    rna_dataset = rna_processor.process(K=N_gene)
    rna_tokens = rna_processor.get_topk_array()
    rna_positions = rna_dataset['gene_position']
    
    atac_processor = PeakProcessor(dataset_path,peak2token_path,ENSG2token_path,peak_cluster_path,idf_path)
    atac_dataset, vocab_size = atac_processor.process(K=N_peak)
    atac_tokens = atac_processor.get_topk_array()
    atac_positions = atac_dataset['atac_position']

    # Concatenate RNA and ATAC data
    combined_data = np.concatenate((rna_tokens, atac_tokens),axis=1)
    return combined_data,rna_positions,atac_positions,vocab_size,test_cell_type,atac_dataset


def hybrid_masking_pair_then_random(rna, atac, mask_ratio=0.15, max_pos_diff=10000):
    """
    rna: (2000, 4), atac: (3000, 4)
    return: final_rna_mask_indices, final_atac_mask_indices
    [id,value,chr,cluster,position,modality]
    """
    N_gene, N_peak = rna.shape[0], atac.shape[0]
    total_tokens = N_gene + N_peak
    total_mask_tokens = int(total_tokens * mask_ratio) 

    rna_chr = rna[:, 2]
    rna_pos = rna[:, 4]
    atac_chr = atac[:, 2]
    atac_pos = atac[:, 4]

    match_chr = rna_chr[:, None] == atac_chr[None, :]
    diff_pos = np.abs(rna_pos[:, None] - atac_pos[None, :])
    matched_mask = match_chr & (diff_pos <= max_pos_diff)

    rna_idx, atac_idx = np.where(matched_mask)
    matched_pairs = list(zip(rna_idx, atac_idx))
    np.random.shuffle(matched_pairs)

    masked_rna = set()
    masked_atac = set()

    if len(matched_pairs) >= total_mask_tokens:
        matched_pairs = matched_pairs[:total_mask_tokens]
        for r_idx, a_idx in matched_pairs:
            if np.random.rand() < 0.5:
                masked_rna.add(r_idx)
            else:
                masked_atac.add(a_idx)
        return sorted(list(masked_rna)), sorted(list(masked_atac))
    
    else:

        for r_idx, a_idx in matched_pairs:
            if np.random.rand() < 0.5:
                masked_rna.add(r_idx)
            else:
                masked_atac.add(a_idx)

        M = len(masked_rna) + len(masked_atac)
        remaining_mask = total_mask_tokens - M

        paired_rna = set(rna_idx)
        paired_atac = set(atac_idx)
        all_rna = set(range(N_gene))
        all_atac = set(range(N_peak))
        nonpair_rna = list(all_rna - paired_rna - masked_rna)
        nonpair_atac = list(all_atac - paired_atac - masked_atac)

        candidates = [(0, i) for i in nonpair_rna] + [(1, i) for i in nonpair_atac]
        np.random.shuffle(candidates)

        for src, idx in candidates:
            if remaining_mask <= 0:
                break
            if src == 0:
                masked_rna.add(idx)
            else:
                masked_atac.add(idx)
            remaining_mask -= 1

        return sorted(list(masked_rna)), sorted(list(masked_atac))
      
def plot_test_result_old(test_cls,test_cell_type,save_dir_result,name):
  cls = test_cls.astype(float)
  cls_norm = np.linalg.norm(cls, 2, axis=1)  # 范数
  cls_cos = cosine_similarity(cls, cls)  # 余弦相似度
  cls_ = sc.AnnData(X=cls)
  cls_.obs_names = [i + "_CLS" for i in cls_.obs_names]
  cls_.obs["domain"] = "CLS"
  cls_.obsm["latent"] = cls_.X
  sc.pp.neighbors(cls_, use_rep="latent", metric="cosine")
  sc.tl.umap(cls_)
  cls_.obs['cell_type'] = test_cell_type
  # 计算指标
  result = calculate_metrics(cls_, name)
  # 保存UMAP图
  sc.settings.set_figure_params(dpi=120, figsize=(10, 5))
  sc.pl.umap(
       cls_,
       color=["cell_type"],
       title='CLS UMAP',
       show=False
          )
  plt.tight_layout()
  plt.savefig(f"{save_dir_result}/{name}_union_embed_umap.png",dpi=300) 
  return result

def plot_test_result(test_cls,test_cell_type,save_dir_result,name):
  cls = test_cls.astype(float)
  #cls_norm = np.linalg.norm(cls, 2, axis=1)  # 范数
  #cls_cos = cosine_similarity(cls, cls)  # 余弦相似度
  cls_ = sc.AnnData(X=cls)
  cls_.obs_names = [i + "_CLS" for i in cls_.obs_names]
  cls_.obs["domain"] = "CLS"
  cls_.obsm["latent"] = cls_.X
  sc.pp.neighbors(cls_, use_rep="latent") # , metric="cosine"
  sc.tl.umap(cls_)
  cls_.obs['cell_type'] = test_cell_type
  # 计算指标
  result = calculate_metrics(cls_, name)
  # 保存UMAP图
  figsize = (13, 5) if name == "03_hJejunum_10xDemo_10x" else (10, 5)
  sc.settings.set_figure_params(dpi=120, figsize=figsize)
  s = 5 if name == "10_hBMMC_10x" else None  # None 表示用 scanpy 默认值
  sc.pl.umap(
       cls_,
       color=["cell_type"],
       title='CLS UMAP',
       show=False,s=s
          )
  plt.tight_layout()
  plt.savefig(f"{save_dir_result}/{name}_union_embed_umap.png",dpi=300) 

  # 再保存一版自定义色彩UMAP图
  # 按 name 设置颜色（长度必须等于 cell_type 的类别数，且顺序与 categories 一致）
  if name == "01_hBrain_10xDemo_10x":
    categories_hBrain = ["astrocyte","chandelier cell","fibroblast","glutamatergic neuron","lamp5 GABAergic cortical interneuron","microglial cell","oligodendrocyte","oligodendrocyte precursor cell","pvalb GABAergic cortical interneuron","sncg GABAergic cortical interneuron","sst GABAergic cortical interneuron","vip GABAergic cortical interneuron"]
    colors_hBrain     = ["#9f79d3","#DE287D","#c49c94","#fc9533","#ffe119","#CB4335","#386CAF","#7FC87F","#008080","#000075","#a65628","#b5bd61"]
    cls_.obs["cell_type"]=pd.Categorical(cls_.obs["cell_type"],categories=categories_hBrain,ordered=True)
    cls_.uns["cell_type_colors"]=colors_hBrain

  elif name == "03_hJejunum_10xDemo_10x":
    categories_hJejunum = ["B cell","T cell","endothelial cell of artery","endothelial cell of lymphatic vessel","enterocyte","fibroblast","glial cell","intestine goblet cell","intestinal crypt stem cell","intestinal enteroendocrine cell","intestinal tuft cell","lymphocyte","macrophage","mesenchymal stem cell","paneth cell","plasma cell","smooth muscle cell","vein endothelial cell"]
    colors_hJejunum = ["#c5b0d5","#fc9533","#76C7C8","#E7E6B0","#7FC87F","#c49c94","#000075","#386CAF","#9f79d3","#DE287D","#ffe119","#008080","#e6194b","#aec7e8","#C29B39","#e377c2","#a65628","#C7E0C9"]
    cls_.obs["cell_type"]=pd.Categorical(cls_.obs["cell_type"],categories=categories_hJejunum,ordered=True)
    cls_.uns["cell_type_colors"]=colors_hJejunum

  elif name == "09_hPBMC_10k_scGLUE_10xDemo":
    categories_hPBMC=["CD14 Mono","CD16 Mono","CD4 Naive","CD4 TCM","CD4 TEM","CD8 Naive","CD8 TEM_1","CD8 TEM_2","HSPC","Intermediate B","MAIT","Memory B","NK","Naive B","Plasma","Treg","cDC","gdT","pDC"]
    colors_hPBMC=["#386CAF","#9f79d3","#aec7e8","#DBDB93","#DE287D","#fc9533","#C29B39","#b5bd61","#a65628","#c5b0d5","#ffe119","#e377c2","#008080","#76C7C8","#3498DB","#000075","#7FC87F","#0082c8","#8E063B"]
    cls_.obs["cell_type"]=pd.Categorical(cls_.obs["cell_type"],categories=categories_hPBMC,ordered=True)
    cls_.uns["cell_type_colors"]=colors_hPBMC

  elif name == "10_hBMMC_10x":
    categories_hBMMC=["B1 B","CD14+ Mono","CD16+ Mono","CD4+ T activated","CD4+ T naive","CD8+ T","CD8+ T naive","cDC2","Erythroblast","G/M prog","HSC","ID2-hi myeloid prog","ILC","Lymph prog","MK/E prog","Naive CD20+ B","NK","Normoblast","pDC","Plasma cell","Proerythroblast","Transitional B"]
    colors_hBMMC=["#163EA1","#7F87B6","#aec7e8","#E7E6B0","#BEC1D3","#386CAF","#b5bd61","#ffe119","#7FC87F","#C29B39","#3498DB","#DE287D","#008080","#76C7C8","#000075","#E7D4C7","#fc9533","#8E063B","#a65628","#e377c2","#9f79d3","#c49c94"]
    cls_.obs["cell_type"]=pd.Categorical(cls_.obs["cell_type"],categories=categories_hBMMC,ordered=True)
    cls_.uns["cell_type_colors"]=colors_hBMMC

  # （可选但建议）检查一下颜色数是否匹配类别数
  n_cat = len(cls_.obs['cell_type'].cat.categories)
  if 'cell_type_colors' in cls_.uns and len(cls_.uns['cell_type_colors']) != n_cat:
    raise ValueError(f"{name}: cell_type_colors({len(cls_.uns['cell_type_colors'])}) != n_categories({n_cat})")
    

  # 保存UMAP图
  figsize = (13, 5) if name == "03_hJejunum_10xDemo_10x" else (10, 5)
  sc.settings.set_figure_params(dpi=300, figsize=figsize)
  s = 5 if name == "10_hBMMC_10x" else None  # None 表示用 scanpy 默认值
  sc.pl.umap(cls_,color=["cell_type"],title=[f"{name} CLS cell type"],show=False,ncols=1,s=s)
  plt.tight_layout()
  plt.savefig(f"{save_dir_result}/{name}_union_embed_umap_custom_colors.png", dpi=300) # bbox_inches="tight",,  pad_inches=0.05
  return result