# Progressive $\ell_p$-Norm Squeezing for Sparse Optimization

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.11+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-GPL-green.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://github.com/WilliamLiPro/lp-squeeze)

> **$\ell_p$ Squeeze**: A soft sparsification method that gradually compresses non-support entries via an $\ell_p$-norm budget, avoiding the premature zeroing-out problem of Iterative Hard Thresholding (IHT).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [TCGA Pan-Cancer Classification](#tcga-pan-cancer-classification)
- [Key Results](#key-results)
- [Citation](#citation)

---
## 🔍 Overview

Sparse optimization underpins compressed sensing, feature selection, and sparse neural networks. Iterative Hard Thresholding (IHT) enforces sparsity via a discontinuous projection that abruptly zeros out small entries, risking premature discarding of significant components.

**$\ell_p$ Squeeze** mitigates this by:

- Replacing the $\ell_0$ constraint with an explicit $\ell_p$-norm budget
- Gradually decreasing $p \to 0_+$ while shrinking the support size
- Progressively evolving the feasible region toward sparse subspaces
- Compressing (rather than zeroing out) non-support entries, allowing them to re-enter the support set in later iterations

On cardinality-constrained least squares where closed-form subproblem solutions exist, $\ell_p$ Squeeze performs comparably to IHT; on **TCGA pan-cancer classification** and **sparse ViT-B/16 training on ImageNet-1K**, it exceeds IHT and its variants with faster convergence and higher accuracy, surpassing existing state-of-the-art approaches.

## 🛠️ Installation

### Prerequisites

- Python >= 3.11
- PyTorch >= 2.11
- CUDA >= 13.0 (for GPU training)

### Setup

```bash
# Clone the repository
git clone https://github.com/WilliamLiPro/lp-squeeze.git
cd lp-squeeze

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
Core dependencies (see requirements.txt for full list):

| Package      | Version   | Purpose                 |
| :----------- |:----------| :---------------------- |
| torch        | \>=2.11.0 | Deep learning framework |
| numpy        | \>=2.4.3  | Numerical computation   |
| pandas       | \>=3.0.2  | Data processing (TCGA)  |
| matplotlib   | \>=3.10.8 | Visualization           |

## 🚀 Quick Start

Run the demo:

```bash
python "Fig2 Comparison of hard thresholding and soft squeezing.py"
```

## 📖 Usage

### TCGA Pan-Cancer Classification

#### Dataset Setup:

The TCGA Pan-Cancer dataset is large (~100MB). We recommend using Git LFS:

```bash
# Install Git LFS (one-time)
git lfs install

# Pull dataset
git lfs pull
```
Or download manually from [TCGA Data Portal](https://www.kaggle.com/datasets/debatreyadas/gene-expression-cancer-rna-seq) and place in `TCGA-PANCAN-HiSeq-801x20531/`.

#### Training:

```bash
python TCGA_Pan_Cancer/tcga_pancan.py
python TCGA_Pan_Cancer/tcga_figure.py
```

## 📊 Key Results

### TCGA Pan-Cancer Classification

We evaluate $\ell_p$ Squeeze on the TCGA Pan-Cancer dataset under varying sparsity levels $\alpha$.

**Convergence Curves:**

<p align="center">
  <img src="figures/tcga_loss_curve_0.0001.png" width="45%" alt="TCGA alpha=1e-4">
</p>

<p align="left"> Convergence curves for varying target non-zero ratios on the TCGA Pan-Cancer classification. $\alpha = 1\times10^{-4}$.</p>

**Final Loss Comparison:**

| Method | $\alpha = 8\times10^{-4}$ | $5\times10^{-4}$ | $2.5\times10^{-4}$ | $1\times10^{-4}$ |
|:---|:---:|:---:|:---:|:---:|
| IHT-AdamW | 1.29e-06 | 5.29e-06 | 3.99e-06 | 7.84e-06 |
| Iterative-HTP-AdamW | 1.29e-06 | 3.95e-06 | 6.14e-06 | 9.65e-06 |
| CGIHT-AdamW | 1.35e-06 | 3.49e-06 | 6.21e-06 | 1.39e-05 |
| FISTA | 1.61e+00 | 1.61e+00 | 1.61e+00 | 1.61e+00 |
| **$\ell_p$ Squeeze-AdamW** | **2.44e-07** | **6.01e-08** | **1.63e-06** | **1.60e-06** |

$\ell_p$ Squeeze-AdamW achieves the **lowest loss across all sparsity levels**.

### Algorithm Mechanism

#### $\ell_p$-Norm Budget Evolution

As $p$ decreases from $2$ to $0_+$, the feasible region progressively converges toward sparse subspaces:

<p align="center">
  <img src="figures/lp_sphere_and_x_path_k1.png" width="80%" alt="k=1">
</p>

<p align="center">
  <img src="figures/lp_sphere_and_x_path_k2.png" width="80%" alt="k=2">
</p>

<p align="left"> Evolution of the feasible set and typical $x$ as $p$ decreases from $2$ to $0_+$. 
(top) $k=1$: the feasible set converges progressively toward the coordinate axes, and $x$ approaches $\|x\|_0=1$. 
(bottom) $k=2$: the feasible set converges progressively toward the three coordinate planes, and $x$ approaches $\|x\|_0=2$.</p>

#### Soft Squeezing vs. Hard Thresholding

<p align="center">
  <img src="figures/mechanism_compare.png" width="80%" alt="Mechanism Comparison">
</p>

<p align="left"> Comparison of hard thresholding and soft squeezing. 
(a) Hard Thresholding: non-support entries within dead zone (red) are immediately zeroed out. 
(b) Soft Squeezing: non-support entries in the squeezing zone (orange) are progressively compressed rather than zeroed out. As $p \to 0_+$, the squeezing operator asymptotically recovers the hard thresholding.</p>

| | Hard Thresholding (IHT) | Soft Squeezing ($\ell_p$ Squeeze) |
|:---|:---|:---|
| **Small entries** | Immediately zeroed out (dead zone) | Progressively compressed (squeezing zone) |
| **Information loss** | Irreversible | Preserved for potential recovery |
| **Support update** | Stagnation until entry hits zero | Continuous refinement |
| **Limit behavior** | — | Recovers hard thresholding as $p \to 0_+$ |

---

#### Algorithm Behavior: $\ell_p$ Squeeze vs. IHT

<p align="center">
  <img src="figures/support_size_and_nonsupport_energy.png" width="45%" alt="Support Size and Non-support Energy">
  <img src="figures/entries_all.png" width="45%" alt="Normalized Absolute Weights">
</p>

<p align="left"> Comparison of $\ell_p$ Squeeze and IHT. 
(left) Support size and non-support energy: $\ell_p$ Squeeze (orange) exhibits a gradual decay in support size and a smooth variation in non-support energy, whereas IHT (blue) enforces a fixed support size and immediately zeros out all non-support energy after the first hard-thresholding projection. 
(right) Normalized absolute weights: IHT updates the support set only when an existing entry decays to nearby zero, causing prolonged stagnation of support set; $\ell_p$ Squeeze replaces a support entry once it falls below the largest non-support entry, enabling continuous refinement.</p>

## 📄 License
This project is licensed under the [License](LICENSE) for details.

## 📚 Citation

If you use this work, please consider citing:

```bibtex
@inproceedings{li2026lp,
  title={Progressive $\ell_p$-Norm Squeezing for Sparse Optimization},
  author={Weipeng Li and Xiaogang Yang},
  booktitle={Proceedings of 2026 CAAI International Conference on Artificial Intelligence},
  year={2026},
  month={10},
  address={Haining, China},
}
