import pickle
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset, load_from_disk
import pandas as pd
import re
import numpy as np
import random
import os
from datasets import disable_caching
#disable_caching()

class PeakProcessor:
    def __init__(self, dataset_path, peak2token_path, gene2token_path, peak_cluster_path, idf_path):
        self.dataset_path = dataset_path  # ★加这一行，保存路径供后面命名输出文件用
        self.dataset = load_from_disk(dataset_path)
        #self.dataset=self.dataset.take(2)
        self.peak2token = pd.read_pickle(peak2token_path)#所有peak id 三百万 str_id:token
        self.gene2token = pd.read_pickle(gene2token_path)

        self.rna_vocab_size = max(self.gene2token.values()) + 1#给peak在基因后连续编码
    

        cluster_df = pd.read_csv(peak_cluster_path)#筛选后的peak 十四万 str_id clsuter 
        self.ref_list = set(self.peak2token['hg38_' + peak_id] for peak_id in cluster_df['peak_id'])#筛选后的peak 十四万，只有id_token

        self.atac_ref_dict = {token: idx + self.rna_vocab_size for idx, token in enumerate(sorted(self.ref_list))}#重新对peak id进行编号，变成 old toekn:new token

        self.vocab_size = max(self.atac_ref_dict.values()) + 1
        
        # Load IDF
        idf_df = pd.read_csv(idf_path, sep="\t")#所有的peak，IDF
        #只选择在self.atac_ref_dict的token进行字典映射，并直接映射为重新编码的token
        self.token2idf = {
            self.atac_ref_dict[self.peak2token['hg38_' + peak]]: idf
            for peak, idf in zip(idf_df["peak"], idf_df["IDF"])
            if self.peak2token['hg38_' + peak] in self.atac_ref_dict
        }

        # Load cluster
        #因为cluster_df都在self.atac_ref_dict，所以直接匹配
        self.token2cluster = {
            self.atac_ref_dict[self.peak2token['hg38_' + peak]]: label
            for peak, label in zip(cluster_df["peak_id"], cluster_df["kmeans_labels"])
            if 'hg38_' + peak in self.peak2token
        }

        # Chromosome encoding
        self.token2chr = {}
        self.token2position = {}
        for peak_str, token_id in self.peak2token.items():
          #过滤未筛选peak
          if token_id in self.atac_ref_dict:
            parts = peak_str.split("_")
            if len(parts) < 2:
                continue
            chr_raw = parts[1][3:]  # "chr1" → "1"
            start_pos = parts[2]
            end_pos = parts[3]
            if chr_raw.isdigit():
                chr_token = int(chr_raw)
            elif chr_raw.upper() == "X":
                chr_token = 23
            elif chr_raw.upper() == "Y":
                chr_token = 24
            else:
                chr_token = 0
            #使用新token进行字典映射
            self.token2chr[self.atac_ref_dict[token_id]] = chr_token
            self.token2position[self.atac_ref_dict[token_id]] = (int(start_pos) + int(end_pos)) // 2

            

    def _filter_peaks(self, example):
        example["raw_peak_num"] = len(example["atac_cell_peaks"]) # 真实peak数（过滤前）
        filtered = [x for x in example["atac_cell_peaks"] if x in self.ref_list]
        example["atac_cell_peaks"] = filtered
        example["peak_num"] = len(filtered)
        return example

    def _add_tfidf(self, example):
        tf = 1.0 / example["peak_num"]
        example["atac_cell_peaks_value"] = [
            tf * self.token2idf.get(x, 0.0) for x in example["atac_cell_peaks"]
        ]
        return example

    def _add_cluster(self, example):
        example["atac_cell_peak_clusters"] = [
            self.token2cluster.get(x, 0) + 2048  for x in example["atac_cell_peaks"]
        ]
        return example

    def _add_chr(self, example):
        example["atac_chr_tokens"] = [
            self.token2chr.get(x, 0) for x in example["atac_cell_peaks"]
        ]
        return example

    def _add_position(self, example):
        example["atac_position"] = [
            self.token2position.get(x, 0) for x in example["atac_cell_peaks"]
        ]
        return example

    def _shift_peak_ids(self, example):
        example["atac_cell_peaks"] = [self.atac_ref_dict[x] for x in example["atac_cell_peaks"]]
        return example


    def save_peak_counts_auto(self, dataset_path, out_dir="peak_num"):
        # dataset_path: .../GRCh38.M00096/RNA_ATAC_data_v1
        sample_name = os.path.basename(os.path.dirname(dataset_path))  # -> GRCh38.M00096
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{sample_name}.txt")
    
        with open(out_path, "w") as f:
            for raw_n, inter_n in zip(self.dataset["raw_peak_num"], self.dataset["peak_num"]):
                f.write(f"{raw_n}\t{inter_n}\n")


    def process(self, K):
        self.dataset = self.dataset.map(self._filter_peaks)

        # 这里保存：每行一个细胞，两列 raw_peak_num / peak_num
        self.save_peak_counts_auto(self.dataset_path, out_dir="peak_num")
    
        self.dataset = self.dataset.map(self._shift_peak_ids)
        self.dataset = self.dataset.map(self._add_tfidf)
        self.dataset = self.dataset.map(self._add_cluster)
        self.dataset = self.dataset.map(self._add_chr)
        self.dataset = self.dataset.map(self._add_position)
        
        self.K = K
        return self.dataset, self.vocab_size

    def get_topk_array(self):
        all_cells = []
        for pid, pval, pclust, pchr, pdis in zip(
            self.dataset["atac_cell_peaks"],
            self.dataset["atac_cell_peaks_value"],
            self.dataset["atac_cell_peak_clusters"],
            self.dataset["atac_chr_tokens"],
            self.dataset["atac_position"]
        ):
            # 给p值生成排名
            pval_np = np.array(pval) # ★
            sort_indices = np.argsort(-pval_np) # ★
            ranks = np.empty_like(sort_indices) # ★
            ranks[sort_indices] = np.arange(1, len(pval) + 1) # ★
            
            peak_set = set(pid)
            feature_num=int(self.K/2)#一半正，一半负
            #正样本充足
            if len(pid) >= feature_num:
                idx = np.random.choice(len(pid), size=feature_num, replace=False)
                idx = idx[np.argsort(-np.array(pval)[idx])]   # 随机抽 K 个 peak 后，把它们按 TF-IDF 从高到低排好顺序。

                # 填充负样本
                sampled_ids = random.sample(list(set(self.atac_ref_dict.values()) - peak_set), (self.K-feature_num))
                neg_ranks = np.random.randint(1, feature_num + 1, size=len(sampled_ids))  # ★随机 1..feature_num
                padded_vectors = [[p, 0.0, self.token2chr.get(p, 0), self.token2cluster.get(p, 0), self.token2position.get(p, 0), 0] for p, r in zip(sampled_ids, neg_ranks)] # ★rank不再设置为0
                vectors  = [[pid[i], pval[i], pchr[i], pclust[i], pdis[i], r+1] for r, i in enumerate(idx)] + padded_vectors
                
            #正样本不够    
            else:
                sampled_ids = random.sample(list(set(self.atac_ref_dict.values()) - peak_set), self.K - len(pid))
                #print(sampled_ids.max())
                # 填充负样本
                max_rank = max(1, len(pid))  # ★用该cell真实peak数做范围（也可以用 feature_num）
                neg_ranks = np.random.randint(1, max_rank + 1, size=len(sampled_ids))  # ★随机 1..len(pid)
                padded_vectors = [[p, 0.0, self.token2chr.get(p, 0), self.token2cluster.get(p, 0), self.token2position.get(p, 0), 0] for p, r in zip(sampled_ids, neg_ranks)]
                idx = np.argsort(-np.array(pval))
                vectors  = [[pid[i], pval[i], pchr[i], pclust[i], pdis[i], ranks[i]] for i in idx]+padded_vectors



            all_cells.append(vectors)

        return np.stack(all_cells)



if __name__ == "__main__":
    processor = PeakProcessor(
    dataset_path="/data1/home/jkchen/slhu/data/AI_Machine_Learning/SCPOE-X/Dataset/scM/processed_data/RNA_ATAC/GRCh38.M00096/RNA_ATAC_data_v1/",
    peak2token_path="/data1/home/jkchen/slhu/data/AI_Machine_Learning/SCPOE-X/SCOPE_X_test/other_file/peak2token_dict.pickle",
    gene2token_path="/data1/home/jkchen/slhu/data/AI_Machine_Learning/SCPOE-X/SCOPE_X_test/other_file/hm_ENSG2token_dict.pickle",
    peak_cluster_path="/data1/home/jkchen/slhu/data/AI_Machine_Learning/SCPOE-X/SCOPE_X_test/other_file/GRCh38_peak_cluster.csv",
    idf_path="/data1/home/jkchen/slhu/data/AI_Machine_Learning/SCPOE-X/SCOPE_X_test/other_file/peak_IDF_variabilityScore_human.txt",
    
    )
    
    print('process')
    dataset, vocab_size = processor.process(K=8000)
    print('step1')
    rna_tokens = processor.get_topk_array()
    print('step2')
    print(rna_tokens.shape)
    




