import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import scoring functions
from anomaly_scores import score_knn, score_lof, score_loci, score_inflo, score_cof, score_ccof

np.random.seed(123)

n_values = [100, 500, 1000, 5000]
p = 10
k = 10
B = 10  # Repetitions

runtime_list = []

for n in n_values:
    print(f"Running n = {n}")
    for b in range(B):
        X = np.random.normal(size=(n, p))
        
        # KNN
        t0 = time.perf_counter()
        _ = score_knn(X, k)
        t1 = time.perf_counter() - t0
        
        # LOF
        t0 = time.perf_counter()
        _ = score_lof(X, k)
        t2 = time.perf_counter() - t0
        
        # LOCI
        t0 = time.perf_counter()
        _ = score_loci(X, k)
        t3 = time.perf_counter() - t0
        
        # INFLO
        t0 = time.perf_counter()
        _ = score_inflo(X, k)
        t4 = time.perf_counter() - t0

        # COF
        t0 = time.perf_counter()
        _ = score_cof(X, k)
        t5 = time.perf_counter() - t0
        
        # CCOF
        t0 = time.perf_counter()
        _ = score_ccof(X, k)
        t6 = time.perf_counter() - t0
        
        runtime_list.extend([
            {'n': n, 'method': 'KNN', 'time': t1},
            {'n': n, 'method': 'LOF', 'time': t2},
            {'n': n, 'method': 'LOCI', 'time': t3},
            {'n': n, 'method': 'INFLO', 'time': t4},
            {'n': n, 'method': 'COF', 'time': t5},
            {'n': n, 'method': 'CCOF', 'time': t6},
        ])

runtime_df = pd.DataFrame(runtime_list)
runtime_avg = runtime_df.groupby(['n', 'method'], as_index=False)['time'].mean()
print("\nAverage Runtime:")
print(runtime_avg)

# Plotting runtime curves
methods = ["KNN", "LOF", "LOCI", "INFLO", "COF", "CCOF"]
markers = ['o', '^', 's', 'D', '*', 'v']
linestyles = ['-', '--', ':', '-.', '-', '--']

plt.figure(figsize=(8, 6))
for i, m in enumerate(methods):
    sub = runtime_avg[runtime_avg['method'] == m]
    plt.plot(sub['n'], sub['time'], label=m, marker=markers[i], linestyle=linestyles[i], color='black', linewidth=2)

plt.xlabel("Sample size (n)", fontsize=12)
plt.ylabel("Runtime (seconds)", fontsize=12)
plt.title("Runtime Comparison across Sample Sizes", fontsize=14)
plt.legend(loc="topleft", frameon=False, fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("runtime_plot.png", dpi=300)
plt.show()