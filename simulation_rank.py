import pandas as pd
import numpy as np

# Import scoring functions from anomaly_scores.py
from anomaly_scores import score_knn, score_lof, score_loci, score_inflo, score_ccof

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


def make_results_table(X, y, k, m_values=[6, 10, 15, 20, 30]):
    knn_s = score_knn(X, k=k)
    lof_s = score_lof(X, k=k)
    loci_s = score_loci(X, k=k)
    inflo_s = score_inflo(X, k=k)
    ccof_s = score_ccof(X, k=k)
    
    out = []
    for m in m_values:
        a = get_metrics(knn_s, y, m)
        b = get_metrics(lof_s, y, m)
        c = get_metrics(loci_s, y, m)
        d = get_metrics(inflo_s, y, m)
        e = get_metrics(ccof_s, y, m)
        
        out.append({
            "m": m,
            "knn_mt": a["mt"], "knn_Re": a["Re"], "knn_RP": a["RP"],
            "lof_mt": b["mt"], "lof_Re": b["Re"], "lof_RP": b["RP"],
            "loci_mt": c["mt"], "loci_Re": c["Re"], "loci_RP": c["RP"],
            "inflo_mt": d["mt"], "inflo_Re": d["Re"], "inflo_RP": d["RP"],
            "ccof_mt": e["mt"], "ccof_Re": e["Re"], "ccof_RP": e["RP"]
        })
        
    return pd.DataFrame(out)


def print_latex_rows(df):
    for _, row in df.iterrows():
        print(
            f"{int(row['m'])} & "
            f"{int(row['knn_mt'])} & {row['knn_Re']:.3f} & {row['knn_RP']:.3f} & "
            f"{int(row['lof_mt'])} & {row['lof_Re']:.3f} & {row['lof_RP']:.3f} & "
            f"{int(row['loci_mt'])} & {row['loci_Re']:.3f} & {row['loci_RP']:.3f} & "
            f"{int(row['inflo_mt'])} & {row['inflo_Re']:.3f} & {row['inflo_RP']:.3f} & "
            f"{int(row['ccof_mt'])} & {row['ccof_Re']:.3f} & {row['ccof_RP']:.3f} \\\\\n\\hline"
        )


if __name__ == "__main__":
    # Load dataset (change to dat1.txt, dat2.txt, or dat3.txt)
    file_path = "dat1.txt"
    dat = pd.read_csv(file_path, sep=r'\s+')
    
    feature_cols = [c for c in dat.columns if c not in ['label']]
    X = dat[feature_cols].values
    y = np.where(dat['label'] == 'o', 1, 0)
    
    # Run evaluations across different k parameters
    for k_val in [5, 10, 25, 50]:
        print(f"\n--- LaTeX Table Output for k = {k_val} ---")
        tab = make_results_table(X, y, k=k_val)
        print_latex_rows(tab)