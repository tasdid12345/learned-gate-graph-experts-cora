"""Cora (Planetoid public split): uniform mixture of SGC-style depth experts (baseline)
vs a cross-validated learned logistic stacking gate over the same experts (improved)."""
import json, os, pickle, sys, time
import numpy as np
import requests
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score

SEEDS = [42, 43, 44]
DEPTHS = [1, 2, 3]                       # propagation depths of the three experts
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]   # regularisation grid (chosen on validation nodes)
BASE = "https://github.com/kimiyoung/planetoid/raw/master/data/"
N_TRAIN, N_VAL = 140, 500                # Planetoid public split sizes
N_CLASSES = 7


def fetch(url, tries=5):
    """GET with timeout=60 and exponential backoff on 429/5xx or network errors."""
    for i in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code != 429 and r.status_code < 500:
                r.raise_for_status()
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    raise RuntimeError(f"download failed: {url}")


def load_cora():
    """Parse Planetoid raw files into row-normalised features, labels, adjacency, test index."""
    raw = {n: pickle.loads(fetch(BASE + f"ind.cora.{n}"), encoding="latin1")
           for n in ["tx", "allx", "ty", "ally", "graph"]}
    test_idx = np.array(fetch(BASE + "ind.cora.test.index").decode().split(), dtype=int)
    # Row r of the stacked matrices holds node row_node[r]; permute so row i is node i.
    X = sp.vstack([raw["allx"], raw["tx"]]).tocsr()
    Y = np.vstack([raw["ally"], raw["ty"]])
    row_node = np.concatenate([np.arange(raw["allx"].shape[0]), test_idx])
    perm = np.argsort(row_node)
    X, Y = X[perm], Y[perm]
    N = X.shape[0]
    # Symmetric binary adjacency without self-loops
    e = np.array([(u, v) for u, vs in raw["graph"].items() for v in vs])
    A = sp.csr_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(N, N))
    A = ((A + A.T) > 0).astype(float)
    A.setdiag(0)
    A.eliminate_zeros()
    # Row-normalise bag-of-words features (as in SGC)
    rs = np.asarray(X.sum(1)).ravel()
    rs[rs == 0] = 1
    X = (sp.diags(1 / rs) @ X).toarray()
    return X, Y.argmax(1), A, test_idx


def normalised_adj(A):
    """Symmetric normalisation D^-1/2 (A+I) D^-1/2."""
    Ah = A + sp.eye(A.shape[0])
    D = sp.diags(np.asarray(Ah.sum(1)).ravel() ** -0.5)
    return (D @ Ah @ D).tocsr()


def expert_features(S, X):
    """Dense blocks S^k X for every depth k = 1..max(DEPTHS)."""
    out, cur = {}, X
    for k in range(1, max(DEPTHS) + 1):
        cur = S @ cur
        out[k] = cur
    return out


def fit_lr(F, y, C, seed):
    return LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(F, y)


def gate_inputs(probs):
    """(n, n_experts, n_classes) probabilities -> (n, n_experts*n_classes) log-probabilities."""
    return np.log(probs + 1e-12).reshape(len(probs), -1)


def run(seed, F, y, tr, va, te):
    """One full pass of both models for a given seed; returns test metrics and timings."""
    # Shared step (untimed): choose expert C by validation accuracy of the uniform average.
    best = None
    for C in C_GRID:
        ms = {k: fit_lr(F[k][tr], y[tr], C, seed) for k in DEPTHS}
        pv = np.mean([ms[k].predict_proba(F[k][va]) for k in DEPTHS], 0)
        acc = accuracy_score(y[va], pv.argmax(1))
        if best is None or acc > best[0]:
            best = (acc, C)
    C_exp = best[1]

    # BASELINE: fixed uniform average of the three expert probabilities (timed).
    t = time.perf_counter()
    ms = {k: fit_lr(F[k][tr], y[tr], C_exp, seed) for k in DEPTHS}
    p_base = np.mean([ms[k].predict_proba(F[k][te]) for k in DEPTHS], 0)
    t_base = time.perf_counter() - t
    pred_base = p_base.argmax(1)

    # IMPROVED: learned logistic gate on out-of-fold expert log-probs (timed).
    t = time.perf_counter()
    oof = np.zeros((len(tr), len(DEPTHS), N_CLASSES))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(tr, y[tr]):
        for j, k in enumerate(DEPTHS):
            oof[b, j] = fit_lr(F[k][tr[a]], y[tr[a]], C_exp, seed).predict_proba(F[k][tr[b]])
    ms = {k: fit_lr(F[k][tr], y[tr], C_exp, seed) for k in DEPTHS}   # full-train experts
    Zva = gate_inputs(np.stack([ms[k].predict_proba(F[k][va]) for k in DEPTHS], 1))
    Zte = gate_inputs(np.stack([ms[k].predict_proba(F[k][te]) for k in DEPTHS], 1))
    Ztr = gate_inputs(oof)
    best_g = None
    for C in C_GRID:  # gate regularisation chosen on validation nodes
        g = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(Ztr, y[tr])
        acc = accuracy_score(y[va], g.predict(Zva))
        if best_g is None or acc > best_g[0]:
            best_g = (acc, C, g)
    pred_imp = best_g[2].predict(Zte)
    t_imp = time.perf_counter() - t

    return dict(acc_base=accuracy_score(y[te], pred_base), f1_base=f1_score(y[te], pred_base, average="macro"),
                acc_imp=accuracy_score(y[te], pred_imp), f1_imp=f1_score(y[te], pred_imp, average="macro"),
                t_base=t_base, t_imp=t_imp, C_exp=C_exp, C_gate=best_g[1])


def main():
    print("Loading Cora from Planetoid raw files...")
    X, y, A, test_idx = load_cora()
    F = expert_features(normalised_adj(A), X)
    tr = np.arange(N_TRAIN)
    va = np.arange(N_TRAIN, N_TRAIN + N_VAL)
    te = np.sort(test_idx)
    print(f"nodes={len(y)} train={len(tr)} val={len(va)} test={len(te)}")

    runs = []
    for s in SEEDS:
        print(f"Seed {s}: fitting baseline and improved models...")
        runs.append(run(s, F, y, tr, va, te))
        print(f"  baseline acc={runs[-1]['acc_base']:.4f}  improved acc={runs[-1]['acc_imp']:.4f}")

    col = lambda k: np.array([r[k] for r in runs], dtype=float)
    acc_b, acc_i = col("acc_base"), col("acc_imp")
    m = {"baseline_score": acc_b.mean(), "improved_score": acc_i.mean(),
         "improvement": acc_i.mean() - acc_b.mean(),
         "baseline_score_std": acc_b.std(), "improved_score_std": acc_i.std(),
         "baseline_f1": col("f1_base").mean(), "improved_f1": col("f1_imp").mean(),
         "baseline_time_s": col("t_base").mean(), "improved_time_s": col("t_imp").mean(),
         "expert_C": col("C_exp").mean(), "gate_C": col("C_gate").mean()}
    m = {k: float(v) for k, v in m.items()}
    if not np.all(np.isfinite(list(m.values()))):
        print("Non-finite metrics; failing.")
        sys.exit(1)

    results = {
        "question": "Can a cross-validated learned gate over propagation-depth graph experts beat a fixed uniform gate for node classification on the Cora citation graph under the standard Planetoid split?",
        "method": "Same three SGC-style depth experts (S^1X, S^2X, S^3X with logistic regression); baseline averages their probabilities uniformly, improved learns a logistic stacking gate on out-of-fold expert log-probabilities from training nodes.",
        "dataset": {"name": "Cora (Planetoid public split)", "source_url": BASE,
                    "n_rows": int(len(y)), "n_train": int(len(tr)), "n_test": int(len(te)),
                    "same_as_paper": False, "baseline_is_papers_method": False},
        "metric_name": "test accuracy",
        "higher_is_better": True,
        "baseline": "Linear GNN in SGC style: fixed uniform average of logistic-regression experts on S^1X, S^2X and S^3X (S = symmetric-normalised adjacency with self-loops).",
        "improved": "Same three experts, but the uniform average is replaced by a logistic-regression stacker trained on 5-fold out-of-fold expert log-probabilities of the training nodes; its regularisation is chosen on validation nodes.",
        "metrics": m,
        "notes": ("Test set used once, for both models. Expert C is chosen on validation accuracy of the uniform average (shared by both models); "
                  "the gate C is chosen on validation nodes as the method specifies, not by CV on training nodes. "
                  "The baseline is deterministic, so its std over seeds is 0; the improved model's seed only changes the out-of-fold folds. "
                  "With 140 training nodes, out-of-fold experts are trained on 112 nodes, a mismatch with the full-train experts used at test time. "
                  "Single dataset, single public split; differences of a few tenths of a percent are within noise."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    vals = [m["baseline_score"], m["improved_score"]]
    ax.bar(["baseline\n(uniform gate)", "improved\n(learned gate)"], vals,
           yerr=[m["baseline_score_std"], m["improved_score_std"]], capsize=4, color=["gray", "tab:blue"])
    ax.set_ylabel("test accuracy")
    ax.set_title("Cora (Planetoid split), mean over seeds")
    ax.set_ylim(min(vals) - 0.05, max(vals) + 0.03)
    fig.tight_layout()
    fig.savefig("figures/accuracy_bar.png", dpi=120)
    print(f"Done. baseline={m['baseline_score']:.4f} improved={m['improved_score']:.4f} -> results.json")


if __name__ == "__main__":
    main()
