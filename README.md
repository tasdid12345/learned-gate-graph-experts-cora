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

## Reference

This project is motivated by the following paper:

> J Frank, D Richerby, A Scherp (2026). *Chimaera: A Mixture-of-Graph-Experts Architecture for Cross-Task and Cross-Dataset Graph Learning*. arXiv preprint arXiv:2609.08709. https://arxiv.org/abs/2609.08709

Paper author: Prof. Ansgar Scherp, Ulm University (ansgar.scherp@uni-ulm.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
