import umap
import pickle
from sklearn.decomposition import PCA
import sys
from sklearn.metrics import silhouette_score, davies_bouldin_score
import hdbscan
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numba
numba.set_num_threads(16)

def get_metrics(features, labels):
    n_noise = np.sum(labels == -1)
    print("Percentage of noise points:", 100*(n_noise/len(labels)))

    mask = labels != -1
    filtered_features = features[mask]
    filtered_labels = labels[mask]

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print("Number of clusters:", n_clusters)
    cluster_sizes = pd.Series(filtered_labels).value_counts().sort_index()
    for cluster_id, size in cluster_sizes.items():
        print("  Cluster {}: {} points".format(cluster_id, size))

    if n_clusters > 1:
        # Silhouette Score (range: -1 to 1, higher is better)
        sil_score = silhouette_score(filtered_features, filtered_labels)
        # Davies–Bouldin Score (lower is better, ideal is close to 0)
        db_score = davies_bouldin_score(filtered_features, filtered_labels)
        print("Silhouette Score:", sil_score)
        print("Davies–Bouldin Score:", db_score)
    return

def get_embedding(features, n_neighbors=15, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_jobs=1)
    embedding = reducer.fit_transform(features)
    return embedding

def plot_umap(labels, embedding, min_cluster_size, min_samples, n_components, n_neighbors):
    title='UMAP mcs: {} ms: {} nc: {} nn: {}'.format(min_cluster_size, min_samples, n_components, n_neighbors)
    plt.figure(figsize=(8, 6))
    if labels is not None:
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=labels, palette='tab10', s=20, linewidth=0)
        plt.legend(title='Cluster ID', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.7)
    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.tight_layout()
    plt.savefig('vary_all/pca0_{}_{}_{}_{}.png'.format(min_cluster_size, min_samples, n_components, n_neighbors), dpi=200)
    return

def main(n_components, n_neighbors, min_samples, min_cluster_size):
    print('==============')
    print('n_components', n_components)
    print('n_neighbors', n_neighbors)
    print('min_samples', min_samples)
    print('min_cluster_size', min_cluster_size)
    print('==============')

    #combined_feats = np.load('/scratch3/BMC/gpu-ghpcs/Rey.Koki/Cloud_Top_Temp/latent_space/combined_feats_deeper.npy')
    combined_feats = np.load('/scratch3/BMC/gpu-ghpcs/Rey.Koki/Cloud_Top_Temp/latent_space/combined_feats.npy')
    #n_pca_components = 50  # usually between 20–100 works well
    n_pca_components = 20  # usually between 20–100 works well
    pca = PCA(n_components=n_pca_components)
    combined_feats = pca.fit_transform(combined_feats)
    with open("pca_{}.pkl".format(n_pca_components), "wb") as f:
        pickle.dump(pca, f)
    umap_2d = get_embedding(combined_feats)
    with open("umap_2d.pkl", "wb") as f:
        pickle.dump(umap_2d, f)
    umap_nd = get_embedding(combined_feats, n_components=n_components, n_neighbors=n_neighbors)
    with open("umap_{}d_{}_pca{}.pkl".format(n_components, n_neighbors, n_pca_components), "wb") as f:
        pickle.dump(umap_nd, f)
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    hdbscan_labels = hdbscan_clusterer.fit_predict(umap_nd)
    with open("hdbscan_labels_{}_{}.pkl".format(min_cluster_size, min_samples), "wb") as f:
        pickle.dump(hdbscan_labels, f)
    plot_umap(hdbscan_labels, umap_2d, min_cluster_size, min_samples, n_components, n_neighbors)
    print('metrics')
    get_metrics(combined_feats, hdbscan_labels)

if __name__ == '__main__':
    n_components = int(sys.argv[1])
    n_neighbors = int(sys.argv[2])
    min_samples = int(sys.argv[3])
    min_cluster_size = int(sys.argv[4])
    main(n_components, n_neighbors, min_samples, min_cluster_size)
