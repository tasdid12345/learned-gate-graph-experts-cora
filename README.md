# Learned Gate over Propagation-Depth Graph Experts on Cora

Motivation: "Chimaera: A Mixture-of-Graph-Experts Architecture for Cross-Task and Cross-Dataset Graph Learning" (2026) combines several graph experts with a gating function so one model can transfer across tasks and datasets. The gate in that work is learned, but its benefit over a fixed mixture is not isolated. This experiment tests the gating idea on a small, controlled setting.

Question: Can a cross-validated learned gate over propagation-depth graph experts beat a fixed uniform gate for node classification on the Cora citation graph under the standard Planetoid split?

Dataset: Cora (2,708 papers, 7 classes), loaded at run time from the Planetoid raw files on GitHub, with the public split of 140 training, 500 validation and 1,000 test nodes. This is not the paper's own dataset.

Baseline: a linear SGC-style GNN. Three logistic-regression experts use features propagated at depths 1, 2 and 3 over the normalised adjacency, and their probabilities are averaged with fixed uniform weights.

Improved model: the same three experts, with the uniform average replaced by a logistic-regression stacker trained on out-of-fold expert probabilities from the training nodes. Its regularisation is chosen on validation nodes. The test set is used once, for both models.

How to run:

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to results.json (and figures/accuracy_bar.png).

## Results (latest run)

On the Cora public Planetoid split, a standard linear GNN with a fixed uniform mixture of experts reached 79.9% test accuracy, while a cross-validated learned stacking gate reached 80.1%, a gain of about 0.2 points that is within seed noise (std 0.3%).

| Metric | Value |
|---|---|
| baseline_score | 0.799 |
| improved_score | 0.8013 |
| improvement | 0.002333 |
| baseline_score_std | 0 |
| improved_score_std | 0.003399 |
| baseline_f1 | 0.7859 |
| improved_f1 | 0.7909 |
| baseline_time_s | 1.164 |
| improved_time_s | 6.047 |
| expert_C | 100 |
| gate_C | 0.1 |

Run on 2026-10-12. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Test set used once, for both models. Expert C is chosen on validation accuracy of the uniform average (shared by both models); the gate C is chosen on validation nodes as the method specifies, not by CV on training nodes. The baseline is deterministic, so its std over seeds is 0; the improved model's seed only changes the out-of-fold folds. With 140 training nodes, out-of-fold experts are trained on 112 nodes, a mismatch with the full-train experts used at test time. Single dataset, single public split; differences of a few tenths of a percent are within noise.

## Reference

This project is motivated by the following paper:

> J Frank, D Richerby, A Scherp (2026). *Chimaera: A Mixture-of-Graph-Experts Architecture for Cross-Task and Cross-Dataset Graph Learning*. arXiv preprint arXiv:2609.08709. https://arxiv.org/abs/2609.08709

Paper author: Prof. Ansgar Scherp, Ulm University (ansgar.scherp@uni-ulm.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
