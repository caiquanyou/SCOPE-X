import scanpy as sc
import anndata as ad
import pandas as pd
import os
import numpy as np
import json
import argparse
from sklearn import metrics
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import connected_components
from sklearn.metrics.cluster import silhouette_samples, silhouette_score
import scib
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    homogeneity_score,
    silhouette_score
)

def integrate(rna_, atac_):
    # Combine GEX & ATAC
    rna_ = sc.AnnData(X=rna_)
    atac_ = sc.AnnData(X=atac_)
    rna_.obs_names = [i + "_RNA" for i in rna_.obs_names]
    atac_.obs_names = [i + "_ATAC" for i in atac_.obs_names]
    rna_.obs["domain"] = "GEX"
    atac_.obs["domain"] = "ATAC"
    combined_ = ad.concat([rna_, atac_])
    combined_.obsm["latent"] = combined_.X
    sc.pp.neighbors(combined_, use_rep="latent", metric="cosine")
    sc.tl.umap(combined_)
    return combined_

def cosine_similarity(A, B):
    """
        计算两个矩阵A和B中每个向量之间的余弦相似度矩阵。

        参数:
        A -- 第一个矩阵，形状为 (m, n)
        B -- 第二个矩阵，形状为 (k, n)

        返回:
        similarity_matrix -- A中每个向量与B中每个向量之间的余弦相似度矩阵，形状为 (m, k)
        """
    # 归一化矩阵
    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=1, keepdims=True)

    # 计算点积
    similarity_matrix = np.dot(A_norm, B_norm.T)

    return similarity_matrix

def asw_batch(
    adata,
    batch_key,
    label_key,
    embed,
    metric="euclidean",
    return_all=False,
    scale=True,
    verbose=True,
):
    """Batch ASW

    Modified average silhouette width (ASW) of batch

    This metric measures the silhouette of a given batch.
    It assumes that a silhouette width close to 0 represents perfect overlap of the batches, thus the absolute value of
    the silhouette width is used to measure how well batches are mixed.
    For all cells :math:`i` of a cell type :math:`C_j`, the batch ASW of that cell type is:
    :param batch_key: batch labels to be compared against
    :param label_key: group labels to be subset by e.g. cell type
    :param embed: name of column in adata.obsm
    :param metric: see sklearn silhouette score
    :param scale: if True, scale between 0 and 1
    :param return_all: if True, return all silhouette scores and label means
        default False: return average width silhouette (ASW)
    :param verbose: print silhouette score per group
    :return:
        Batch ASW 
        Mean silhouette per group in pd.DataFrame (additionally, if return_all=True)
        Absolute silhouette scores per group label (additionally, if return_all=True)

    The function requires an embedding to be stored in ``adata.obsm`` and can only be applied to feature and embedding
    integration outputs.
    Please note, that the metric cannot be used to evaluate kNN graph outputs.
    See :ref:`preprocessing` for more information on preprocessing.

    **Examples**
        asw_batch = asw_batch(
            adata,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            metric=si_metric,
            return_all=False,
            verbose=False,
        )
    """
    if embed not in adata.obsm.keys():
        print(adata.obsm.keys())
        raise KeyError(f"{embed} not in obsm")

    sil_per_label = []
    for group in adata.obs[label_key].unique():
        adata_group = adata[adata.obs[label_key] == group]
        n_batches = adata_group.obs[batch_key].nunique()

        if (n_batches == 1) or (n_batches == adata_group.shape[0]):
            continue

        sil = silhouette_samples(
            adata_group.obsm[embed], adata_group.obs[batch_key], metric=metric
        )

        # take only absolute value
        sil = [abs(i) for i in sil]

        if scale:
            # scale s.t. highest number is optimal
            sil = [1 - i for i in sil]

        sil_per_label.extend([(group, score) for score in sil])

    sil_df = pd.DataFrame.from_records(
        sil_per_label, columns=["group", "silhouette_score"]
    )

    if len(sil_per_label) == 0:
        sil_means = np.nan
        asw = np.nan
    else:
        sil_means = sil_df.groupby("group").mean()
        asw = sil_means["silhouette_score"].mean()

    if verbose:
        print(f"mean silhouette per group: {sil_means}")

    if return_all:
        return asw, sil_means, sil_df

    return asw

def graph_connectivity(adata, label_key):
    """Graph Connectivity
    Quantify the connectivity of the subgraph per cell type label.
    The final score is the average for all cell type labels :math:`C`, according to the equation:
    """

    adata.obs[label_key] = adata.obs[label_key].astype('category')
    clust_res = []
    for label in adata.obs[label_key].cat.categories:
        adata_sub = adata[adata.obs[label_key].isin([label])]
        _, labels = connected_components(
            adata_sub.obsp["connectivities"], connection="strong"
        )
        tab = pd.value_counts(labels)
        clust_res.append(tab.max() / sum(tab))

    return np.mean(clust_res)

def calculate_metrics(combined,dataset_name):

    labels = combined.obs["cell_type"].values
    # Automatically select best resolution for clustering
    cluster_key_temp = "cluster_temp"
    combined.obs[cluster_key_temp] = "-1"
    sc.tl.leiden(combined, resolution=1, key_added="predicted_cluster")
    clusters = combined.obs["predicted_cluster"].values
    ari = adjusted_rand_score(labels, clusters)
    ami = adjusted_mutual_info_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)
    hom = homogeneity_score(labels, clusters)
    graph_conn = graph_connectivity(combined, label_key="cell_type")
    asw = silhouette_score(combined.obsm["X_umap"], labels)
    avgbio = (nmi + ari + asw) / 3
    
    result = {
        "dataset": dataset_name,
        "ARI": round(float(ari), 4),
        "AMI": round(float(ami), 4),
        "NMI": round(float(nmi), 4),
        "ASW": round(float(asw), 4) if not np.isnan(asw) else None,
        "Homogeneity": round(float(hom), 4),
        "AvgBio": round(float(avgbio), 4) if not np.isnan(avgbio) else None,
        'graph_connectivity':round(float(graph_conn), 3),
    }
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str,
                        default='/media/GPU_Storage/zhaoyy/scFM_data_peak_filter/evaluation_data/PBMC/raw_h5ad/PBMC_rna_raw.h5ad')
    parser.add_argument("--embed_path", type=str,
                        default='/home/zhaoyy/ia_lgl/code8/down_stream_outputs/get_embeds/RNA_ATAC_mergedCks_merge1_tvt/PBMC_RNA_bs32_directOut1/')
    parser.add_argument("--save_path", type=str,
                        default='results')
    
    args = parser.parse_args()

    data_path = args.embed_path
    save_path = os.path.join(data_path, args.save_path) 
    os.makedirs(save_path, exist_ok=True)
    # # For Zeroshot
    raw_adata = sc.read_h5ad(args.raw)
    rna = np.load(os.path.join(data_path, 'rna_cell_embs.npy'))
    atac = np.load(os.path.join(data_path, 'atac_cell_embs.npy'))
    cell_names = np.load(os.path.join(data_path, 'cell_names.npy'))
    raw_adata = raw_adata[cell_names].copy()
    new_obs = pd.concat([raw_adata.obs, raw_adata.obs], axis=0)

    rna_norm = np.linalg.norm(rna[:10], 2, axis=1)
    atac_norm = np.linalg.norm(atac[:10], 2, axis=1)

    rna_cos = cosine_similarity(rna[:10], rna[:10])
    atac_cos = cosine_similarity(atac[:10], atac[:10])
    rna_atac_cos = cosine_similarity(rna[:10], atac[:10])

    combined = integrate(rna, atac)
    combined.obs['cell_type'] = new_obs['cell_type'].values
    metrics = calculate_metrics(combined, save_path)
    obsm = combined.obsm["X_umap"]
    sc.settings.set_figure_params(dpi=120, figsize=(10, 5))
    sc.pl.umap(
            combined,
            color=["domain","cell_type"],
            title='Domain UMAP (GEX vs ATAC)',
            show=False
        )
    plt.tight_layout()
    plt.savefig(f"{save_path}/union_embed_umap.pdf")