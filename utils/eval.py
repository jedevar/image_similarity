import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    silhouette_score,
)
from sklearn.metrics import pairwise_distances
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment


def evaluate_clustering(true_labels, cluster_ids, features=None, silhouette_sample=5000, verbose=True):
    """
    Evaluate clustering against ground-truth labels.

    Arguments:
      - true_labels: 1D array-like of ints (length N)
      - cluster_ids: 1D array-like of ints (length N)
      - features: optional (N, D) array for silhouette score (if provided)
      - silhouette_sample: max number of points to use for silhouette (randomly sampled)
      - verbose: print a short report

    Returns: dict with metrics and helpful tables
    """
    y = np.asarray(true_labels).astype(int)
    c = np.asarray(cluster_ids).astype(int)
    assert y.shape[0] == c.shape[0], "labels and clusters must have same length"
    N = y.shape[0]

    # 1) contingency matrix: rows = clusters, cols = classes
    cm = contingency_matrix(y_true=y, y_pred=c)  # shape (n_classes, n_clusters)
    # Note sklearn's contingency_matrix uses rows = true classes, cols = pred clusters.
    # We'll build cluster x class matrix for convenience:
    cluster_labels = np.unique(c)
    class_labels = np.unique(y)
    # But we can work with cm as is: cm[class, cluster]
    # Purity (majority per cluster): for each column (cluster), take max over rows (classes)
    cm_T = cm.T  # now shape (n_clusters, n_classes)

    # Purity
    majority_counts = cm_T.max(axis=1)  # per cluster
    purity = majority_counts.sum() / N

    # 2) Hungarian assignment to maximize correct matches
    # We want a square cost matrix for linear_sum_assignment. Use negated counts (maximize).
    n_clusters = cm_T.shape[0]
    n_classes = cm_T.shape[1]
    K = max(n_clusters, n_classes)
    cost = np.zeros((K, K), dtype=np.float32)
    cost[:n_clusters, :n_classes] = -cm_T  # negative because we want to maximize counts
    row_ind, col_ind = linear_sum_assignment(cost)
    # Keep only assignments that correspond to real clusters/classes
    assigned_pairs = [(r, c_) for r, c_ in zip(row_ind, col_ind) if r < n_clusters and c_ < n_classes]
    # matched count:
    matched = sum(cm_T[r, c_] for r, c_ in assigned_pairs)
    hungarian_accuracy = matched / N

    # 3) ARI, NMI, Fowlkes-Mallows
    ari = adjusted_rand_score(y, c)
    nmi = normalized_mutual_info_score(y, c, average_method='arithmetic')  # or 'geometric'
    fm = fowlkes_mallows_score(y, c)

    # 4) silhouette (optional, only if features provided)
    sil = None
    if features is not None:
        # sample if too large
        X = np.asarray(features)
        if X.shape[0] > silhouette_sample:
            idx = np.random.choice(X.shape[0], silhouette_sample, replace=False)
            sil = silhouette_score(X[idx], c[idx], metric='euclidean')
        else:
            sil = silhouette_score(X, c, metric='euclidean')

    # 5) per-cluster table: size, top class, purity
    per_cluster = []
    for cluster_idx in range(cm_T.shape[0]):
        counts = cm_T[cluster_idx]
        size = counts.sum()
        if size == 0:
            top_class = None
            top_count = 0
            purity_k = 0.0
        else:
            top_class = class_labels[counts.argmax()]  # note: class_labels is sorted unique of y
            top_count = counts.max()
            purity_k = top_count / size
        per_cluster.append({
            'cluster': int(cluster_idx),
            'size': int(size),
            'top_class': int(top_class) if top_class is not None else None,
            'top_count': int(top_count),
            'purity': float(purity_k)
        })

    results = {
        'N': int(N),
        'n_clusters': int(n_clusters),
        'n_classes': int(n_classes),
        'purity': float(purity),
        'hungarian_accuracy': float(hungarian_accuracy),
        'ari': float(ari),
        'nmi': float(nmi),
        'fowlkes_mallows': float(fm),
        'silhouette': float(sil) if sil is not None else None,
        'contingency_matrix_class_by_cluster': cm,  # rows=classes, cols=clusters
        'per_cluster': per_cluster,
        'hungarian_assignment': assigned_pairs,  # list of (cluster_idx, class_idx) pairs
    }

    if verbose:
        print(f"N={N} clusters={n_clusters} classes={n_classes}")
        print(f"Purity: {purity:.4f}")
        print(f"Hungarian accuracy: {hungarian_accuracy:.4f}")
        print(f"ARI: {ari:.4f}  NMI: {nmi:.4f}  F-M: {fm:.4f}")
        if sil is not None:
            print(f"Silhouette (sampled up to {silhouette_sample}): {sil:.4f}")
        print("Top clusters by size (cluster, size, top_class, purity):")
        for row in sorted(per_cluster, key=lambda r: r['size'], reverse=True)[:10]:
            print(row)

    return results
