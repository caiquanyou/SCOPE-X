import pickle
import pyarrow as pa
import pyarrow.ipc as ipc
from datasets import Dataset, load_from_disk
import pandas as pd
import re
import numpy as np
import random
from datasets import disable_caching
#disable_caching()

class GeneProcessor:
    def __init__(self, dataset_path, gene2token_path, gene_cluster_path,position_path):
        self.dataset = load_from_disk(dataset_path)
        #self.dataset=self.dataset.take(20)
        self.gene2token = pd.read_pickle(gene2token_path)

        cluster_df = pd.read_csv(gene_cluster_path,header=0, names=['gene_id', 'cluster'])
        position_df = pd.read_csv(position_path, sep='\t', header=None, names=['chromosome', 'start', 'end', 'gene_id', 'score', 'strand'])
        self.ref_list = set(self.gene2token[gene_id] for gene_id in cluster_df['gene_id'])#保留所有涉及到的token


        #将token和cluster对应
        self.token2cluster = {
            self.gene2token[gene]: label
            for gene, label in zip(cluster_df["gene_id"], cluster_df["cluster"])
            if gene in self.gene2token
        }
      
     
        # Chromosome encoding
        self.token2chr = {}
        self.token2position = {}
        for gene, chr,postion in zip(position_df['gene_id'],position_df['chromosome'],position_df['start']):
            if chr.startswith('chr'):
                chr_raw = chr[3:]
            if chr_raw.isdigit():
                chr_token = int(chr_raw)
            elif chr_raw.upper() == "X":
                chr_token = 23
            elif chr_raw.upper() == "Y":
                chr_token = 24
            else:
                chr_token = 0
            self.token2chr[self.gene2token[gene]] = chr_token
            self.token2position[self.gene2token[gene]] = postion

    def _filter_genes(self, example):
        filtered = [x for x in example["rna_gene_ids"] if x in self.ref_list]
        example["rna_gene_ids_filtered"] = filtered
        example["gene_num"] = len(filtered)
        return example
    
    def _add_gene_value(self, example):
        filtered_value = [example["rna_gene_values"][i] for i in range(len(example["rna_gene_ids"])) if example["rna_gene_ids"][i] in example["rna_gene_ids_filtered"]]
        example["rna_gene_value_filtered"] = filtered_value
        
        return example

    def _add_cluster(self, example):
        example["rna_gene_clusters"] = [
            self.token2cluster.get(x, 0) for x in example["rna_gene_ids"]
        ]
        return example

    def _add_chr(self, example):
        example["gene_chr_tokens"] = [
            self.token2chr.get(x, 0) for x in example["rna_gene_ids"]
        ]
        return example

    def _add_position(self, example):
        example["gene_position"] = [
            self.token2position.get(x, 0) for x in example["rna_gene_ids"]
        ]
        return example


    def process(self, K):
        self.dataset = self.dataset.map(self._filter_genes)
        self.dataset = self.dataset.map(self._add_gene_value)
        self.dataset = self.dataset.map(self._add_cluster)
        self.dataset = self.dataset.map(self._add_chr)
        self.dataset = self.dataset.map(self._add_position)
        self.K = K
        return self.dataset

    def get_topk_array(self):
        all_cells = []
        for pid, pval, pclust, pchr, pdis in zip(
            self.dataset["rna_gene_ids_filtered"],
            self.dataset["rna_gene_value_filtered"],
            self.dataset["rna_gene_clusters"],
            self.dataset["gene_chr_tokens"],
            self.dataset["gene_position"]
        ):
            pval_np = np.array(pval)
            sort_indices = np.argsort(-pval_np)
            ranks = np.empty_like(sort_indices)
            ranks[sort_indices] = np.arange(1, len(pval) + 1)
            RNA_set = set(pid)
            if len(pid) >= self.K:
                idx = np.argsort(-np.array(pval))[:self.K]
                vectors = [[pid[i], pval[i], pchr[i], pclust[i], pdis[i], ranks[i]] for i in idx]
            else:
                sampled_ids = random.sample(list(self.ref_list - RNA_set), self.K - len(pid))
                # 填充负样本
                padded_vectors = [[p, 0.0, self.token2chr.get(p, 0), self.token2cluster.get(p, 0), self.token2position.get(p, 0),0] for p in sampled_ids]
                idx = np.argsort(-np.array(pval))
                vectors = [[pid[i], pval[i], pchr[i], pclust[i], pdis[i], ranks[i]] for i in idx] + padded_vectors
            all_cells.append(vectors)

        return np.stack(all_cells)



if __name__ == "__main__":

    processor = GeneProcessor(
    dataset_path="/home/guest/Downloads/SingleCell/dataset/GRCh38.M00007/RNA_ATAC_data_v1/",
    gene2token_path="/home/guest/Downloads/SingleCell/hm_ENSG2token_dict.pickle",
    gene_cluster_path="/home/guest/Downloads/SingleCell/GRCh38_gene_cluster.csv",
    position_path="/home/guest/Downloads/SingleCell/GRCh38.tss.bed")
    print('process')
    dataset = processor.process(K=2000)
    print('step1')
    rna_tokens = processor.get_topk_array()
    print('step2')
    print(rna_tokens.shape)
    