import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra, connected_components



def score_knn(X, k):
    """
    k-Nearest Neighbors Anomaly Score.
    Returns the mean distance to the k-nearest neighbors.
    """
    X = np.asarray(X, dtype=float)
    if k >= len(X):
        raise ValueError("k must be smaller than the number of observations.")
    
    D = dist.squareform(dist.pdist(X))
    idx = np.argsort(D, axis=1)[:, 1:k+1]
    dists = np.take_along_axis(D, idx, axis=1)
    return np.mean(dists, axis=1)


def score_lof(X, k):
    """
    Local Outlier Factor (LOF) Anomaly Score.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if k >= n:
        raise ValueError("k must be smaller than the number of observations.")
    
    D = dist.squareform(dist.pdist(X))
    idx = np.argsort(D, axis=1)[:, 1:k+1]
    dists = np.take_along_axis(D, idx, axis=1)
    k_distances = dists[:, -1]
    
    lrd = np.zeros(n)
    for i in range(n):
        nbrs = idx[i]
        reach_dists = np.maximum(k_distances[nbrs], D[i, nbrs])
        lrd[i] = k / np.sum(reach_dists)
        
    scores = np.zeros(n)
    for i in range(n):
        nbrs = idx[i]
        scores[i] = np.mean(lrd[nbrs]) / (lrd[i] + 1e-12)
        
    return scores


def score_loci(X, k):
    """
    Local Correlation Integral (LOCI) Anomaly Score.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if k >= n:
        raise ValueError("k must be smaller than the number of observations.")
    
    D = dist.squareform(dist.pdist(X))
    alpha = 0.5
    scores = np.zeros(n)
    
    for i in range(n):
        d_i = np.sort(D[i])
        r = d_i[k]  # Distance to k-th nearest neighbor (d_i[0] is self distance)
        
        n_i = np.sum(D[i] <= alpha * r)
        nbrs = np.where(D[i] <= r)[0]
        
        neigh_counts = [np.sum(D[j] <= alpha * r) for j in nbrs]
        mu = np.mean(neigh_counts) if len(neigh_counts) > 0 else 0.0
        
        scores[i] = np.abs(1.0 - n_i / mu) if mu > 0 else 0.0
        
    return scores


def score_inflo(X, k):
    """
    Influenced Outlierness (INFLO) Anomaly Score.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    if k >= n:
        raise ValueError("k must be smaller than the number of observations.")
    
    D = dist.squareform(dist.pdist(X))
    idx = np.argsort(D, axis=1)[:, 1:k+1]
    dists = np.take_along_axis(D, idx, axis=1)
    
    rnn = [[] for _ in range(n)]
    for i in range(n):
        for j in idx[i]:
            rnn[j].append(i)
            
    density = 1.0 / np.mean(dists, axis=1)
    inflo = np.zeros(n)
    
    for i in range(n):
        S = list(set(idx[i]).union(set(rnn[i])))
        if len(S) == 0:
            inflo[i] = 1.0
        else:
            inflo[i] = np.mean(density[S]) / density[i]
            
    return inflo


# INLINE COF IMPLEMENTATION (Connectivity-Based Outlier Factor)
# =====================================================================

def score_cof(X, k):
    """
    Connectivity-Based Outlier Factor (COF).
    Calculates average chaining distance via Set-Based Nearest Neighbor (SBN) sequence.
    """
    X = np.asarray(X, dtype=float)
    n = len(X)
    D = cdist(X, X)
    
    # Identify k-nearest neighbors (excluding self)
    knn_idx = np.argsort(D, axis=1)[:, 1:k+1]
    
    ac_dist = np.zeros(n)
    for i in range(n):
        sbn = [i]
        candidates = list(knn_idx[i])
        cost_sum = 0.0
        
        # Build SBN path chain
        for step in range(1, k + 1):
            min_d = np.inf
            best_cand = None
            for cand in candidates:
                d_to_set = np.min([D[cand, s] for s in sbn])
                if d_to_set < min_d:
                    min_d = d_to_set
                    best_cand = cand
            sbn.append(best_cand)
            candidates.remove(best_cand)
            cost_sum += (2 * (k - step + 1)) * min_d
            
        ac_dist[i] = cost_sum / (k * (k + 1))
        
    cof_scores = np.zeros(n)
    for i in range(n):
        nbrs = knn_idx[i]
        denom = np.mean(ac_dist[nbrs])
        cof_scores[i] = ac_dist[i] / (denom + 1e-8)
        
    return cof_scores



## CCOF IMPLEMENTATION (Connectivity-Based Clustering Outlier Factor)
def score_ccof(X, k):
    X = np.asarray(X, dtype=float)
    n = len(X)
    
    # 1. Spatial Tree k-NN Search O(N log N) - Bypasses dense N x N distance matrix
    tree = cKDTree(X)
    dists, indices = tree.query(X, k=k+1)
    idx = indices[:, 1:]
    knn_dists = dists[:, 1:]
    
    # 2. Vectorized Sparse Mutual k-NN Graph Construction
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    weights = knn_dists.ravel()
    
    adj = csr_matrix((weights, (rows, cols)), shape=(n, n))
    
    # Enforce strict reciprocity in C
    adj_T = adj.T
    mutual_mask = (adj > 0).multiply(adj_T > 0)
    adj_mutual = adj.minimum(adj_T).multiply(mutual_mask)
    
    # 3. C-Accelerated Connected Components
    n_components, labels = connected_components(csgraph=adj_mutual, directed=False)
    comp_sizes = np.bincount(labels)
    cmax = np.max(comp_sizes)
    
    # 4. C-Accelerated Dijkstra Shortest Paths
    dist_matrix = dijkstra(csgraph=adj_mutual, directed=False, return_predecessors=False)
    
    # Penalty for disconnected nodes
    max_d = np.max(knn_dists) * 5.0
    dist_matrix[np.isinf(dist_matrix)] = max_d
    
    # 5. Vectorized Outlier Scoring
    nbr_paths = np.take_along_axis(dist_matrix, idx, axis=1)
    chain = np.mean(nbr_paths, axis=1)
    
    denom = np.mean(chain[idx], axis=1)
    local_part = np.where((np.isfinite(denom) & (denom > 0)), chain / denom, 1.0)
    global_part = cmax / comp_sizes[labels]
    
    return local_part * global_part


def compute_ccof(X, k):
    """
    Wrapper around score_ccof(X, k) that returns CCOF anomaly scores alongside 
    connected component size metadata for simulation inspection.
    """
    # 1. Compute CCOF scores using score_ccof
    ccof_scores = score_ccof(X, k)
    
    # 2. Extract graph component statistics matching score_ccof topology
    X_arr = np.asarray(X, dtype=float)
    n = len(X_arr)
    D = dist.squareform(dist.pdist(X_arr))
    idx = np.argsort(D, axis=1)[:, 1:k+1]
    
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for v in idx[i]:
            G.add_edge(i, v)
            
    comps = list(nx.connected_components(G))
    comp_sizes = {node: len(c) for c in comps for node in c}
    max_comp_size = max(comp_sizes.values()) if comp_sizes else 0
    
    return ccof_scores, comp_sizes, max_comp_size


def get_metrics(score, y, m):
    """
    Computes mt (top m anomaly count), Re (recall), and RP (rank power).
    """
    ord_idx = np.argsort(-score)  # Descending order
    topm = ord_idx[:m]
    
    mt = np.sum(y[topm] == 1)
    Re = mt / np.sum(y == 1)
    
    # Rank-power calculation (1-indexed ranks)
    anomaly_indices = np.where(y == 1)[0]
    ranks = [np.where(ord_idx == a_idx)[0][0] + 1 for a_idx in anomaly_indices]
    RP = np.mean((len(y) - np.array(ranks) + 1) / len(y))
    
    return {"mt": mt, "Re": Re, "RP": RP}

def make_results_table(X, y, k, m_values):
    knn_s = score_knn(X, k=k)
    lof_s = score_lof(X, k=k)
    loci_s = score_loci(X, k=k)
    inflo_s = score_inflo(X, k=k)
    cof_s = score_cof(X, k=k)
    ccof_s = score_ccof(X, k=k)
    
    out = []
    for m in m_values:
        a = get_metrics(knn_s, y, m)
        b = get_metrics(lof_s, y, m)
        c = get_metrics(loci_s, y, m)
        d = get_metrics(inflo_s, y, m)
        e = get_metrics(cof_s, y, m)
        f = get_metrics(ccof_s, y, m)
        
        out.append({
            "m": m,
            "knn_mt": a["mt"], "knn_Re": a["Re"], "knn_RP": a["RP"],
            "lof_mt": b["mt"], "lof_Re": b["Re"], "lof_RP": b["RP"],
            "loci_mt": c["mt"], "loci_Re": c["Re"], "loci_RP": c["RP"],
            "inflo_mt": d["mt"], "inflo_Re": d["Re"], "inflo_RP": d["RP"],
            "cof_mt": e["mt"], "cof_Re": e["Re"], "cof_RP": e["RP"],
            "ccof_mt": f["mt"], "ccof_Re": f["Re"], "ccof_RP": f["RP"]
        })
        
    return pd.DataFrame(out)
