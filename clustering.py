import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict

def get_clusters(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    weights = kmeans.fit_predict(data)
    unique, cluster_weights = np.unique(weights, return_counts=True)

    count_dict = defaultdict(int)
    count_dict.update(dict(zip(unique, cluster_weights)))

    cluster_weights = [count_dict[i] for i in range(n_clusters)]

    centers = kmeans.cluster_centers_
    
    return centers, cluster_weights
