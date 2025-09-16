import umap
import hdbscan
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def get_embedding(features, n_neighbors=15, min_dist=0.1, n_components=2):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_jobs=1)
    embedding = reducer.fit_transform(features)
    return embedding

def plot_umap(labels, embedding, cluster_algorithm=None):

    title="UMAP of {} Patch Features".format(cluster_algorithm)

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
    plt.savefig('umap_{}.png'.format(cluster_algorithm), dpi=300)
    return

combined_feats = np.load('/scratch3/BMC/gpu-ghpcs/Rey.Koki/Cloud_Top_Temp/latent_space/combined_feats.npy')

umap_10d = get_embedding(combined_feats, n_components=10)
hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=5)
hdbscan_labels = hdbscan_clusterer.fit_predict(umap_10d)
umap_2d = get_embedding(combined_feats)

plot_umap(hdbscan_labels, umap_2d, cluster_algorithm='HDBSCAN')

kmeans = KMeans(n_clusters=8)
kmeans_labels = kmeans.fit_predict(combined_feats)


plot_umap(kmeans_labels, cluster_algorithm='KMeans')

