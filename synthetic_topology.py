# =====================================================================
# 1. SYNTHETIC TOPOLOGY GENERATORS
# =====================================================================

import numpy as np


def generate_global_local_boundary(seed=42):
    """Topology 1: Main Gaussian cluster with boundary and global outliers."""
    np.random.seed(seed)
    # Main normal cluster
    X_normal = np.random.normal(loc=0.0, scale=1.0, size=(250, 2))
    # Global anomalies
    X_global = np.random.uniform(low=-8.0, high=8.0, size=(10, 2))
    # Filter to keep true global anomalies far from origin
    dist_from_center = np.linalg.norm(X_global, axis=1)
    X_global = X_global[dist_from_center > 5.0]
    
    X = np.vstack([X_normal, X_global])
    y = np.array([0] * len(X_normal) + [1] * len(X_global))
    return X, y


def generate_non_convex_manifolds(seed=42):
    """Topology 2: Non-convex spiral / half-moon geometry with anomalies."""
    np.random.seed(seed)
    n_samples = 150
    
    # Half-moon 1
    t1 = np.linspace(0, np.pi, n_samples)
    x1 = np.cos(t1) + np.random.normal(0, 0.08, n_samples)
    y1 = np.sin(t1) + np.random.normal(0, 0.08, n_samples)
    
    # Half-moon 2
    t2 = np.linspace(0, np.pi, n_samples)
    x2 = 1.0 - np.cos(t2) + np.random.normal(0, 0.08, n_samples)
    y2 = 0.5 - np.sin(t2) + np.random.normal(0, 0.08, n_samples)
    
    X_normal = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    
    # Anomalies placed near center / between moons
    X_anomalies = np.array([
        [0.5, 0.2], [0.5, -0.2], [-0.5, 0.5], 
        [1.5, 0.0], [0.0, 1.2], [1.0, -0.8]
    ])
    
    X = np.vstack([X_normal, X_anomalies])
    y = np.array([0] * len(X_normal) + [1] * len(X_anomalies))
    return X, y


def generate_multi_density_clusters(seed=42):
    """Topology 3: High-density and low-density clusters with anomalies."""
    np.random.seed(seed)
    # Dense cluster
    X_dense = np.random.normal(loc=[-4.0, 0.0], scale=0.4, size=(200, 2))
    # Sparse cluster
    X_sparse = np.random.normal(loc=[4.0, 0.0], scale=1.8, size=(100, 2))
    # Isolated anomalies
    X_anomalies = np.array([
        [0.0, 0.0], [0.0, 3.0], [0.0, -3.0], 
        [-8.0, 0.0], [9.0, 0.0], [4.0, 6.0]
    ])
    
    X = np.vstack([X_dense, X_sparse, X_anomalies])
    y = np.array([0] * (len(X_dense) + len(X_sparse)) + [1] * len(X_anomalies))
    return X, y


def generate_micro_clusters(seed=42):
    """Topology 4: Small collective anomaly / micro-cluster (S <= k)."""
    np.random.seed(seed)
    # Primary cluster (N = 300)
    X_main = np.random.normal(loc=0.0, scale=1.0, size=(300, 2))
    # Micro-cluster anomaly (S = 4)
    X_micro = np.random.normal(loc=[12.0, 12.0], scale=0.1, size=(4, 2))
    
    X = np.vstack([X_main, X_micro])
    y = np.array([0] * len(X_main) + [1] * len(X_micro))
    return X, y

