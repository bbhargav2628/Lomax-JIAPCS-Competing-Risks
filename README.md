# Classical and Bayesian Estimation of the Lomax Distribution under Joint Improved Adaptive Progressive Censored Competing Risks Data

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![Manuscript: 14 Pages](https://img.shields.io/badge/Manuscript-14%20Pages%20Verified-brightgreen.svg)](Lomax_JIAPCS_Project_Report.pdf)
[![Presentation: 15 Slides](https://img.shields.io/badge/Presentation-15%20Slides-orange.svg)](Lomax_JIAPCS_Presentation.pptx)
[![IIT Guwahati](https://img.shields.io/badge/IIT%20Guwahati-DA2141-red.svg)](https://www.iitg.ac.in/)

**Course:** DA2141: Statistics for Data Science — Course Project  
**Institution:** Mehta Family School of Data Science and Artificial Intelligence, Indian Institute of Technology Guwahati (IIT Guwahati), India  
**Authors:**  
* **Ruttala Vamsi Krishna** (Roll No: 250150034)  
* **Bagadi Bhargav** (Roll No: 250150009)  

---

## 📑 Table of Contents
1. [Executive Summary & Abstract](#executive-summary--abstract)
2. [Project Repository Architecture](#project-repository-architecture)
3. [Key Theoretical and Methodological Contributions](#key-theoretical-and-methodological-contributions)
4. [Simulation Study (1000 Replications across 22 Scenarios)](#simulation-study)
5. [Real Data Applications](#real-data-applications)
6. [Quick Start & Reproducibility Guide](#quick-start--reproducibility-guide)
7. [LaTeX Manuscript Compilation](#latex-manuscript-compilation)
8. [Academic Notice & Citation](#academic-notice--citation)

---

## 🔬 Executive Summary & Abstract

This repository contains the complete theoretical derivations, Monte Carlo simulations, real-data analyses, and publication-ready LaTeX manuscript for classical and Bayesian inference of two Lomax populations under a **Joint Improved Adaptive Progressive Type-II Censoring Scheme (JIAPCS)** with two competing causes of failure per population.

```
       ┌────────────────────────────────────────────────────────┐
       │     Two Populations Tested Jointly:  Ω_1 and Ω_2       │
       │    Each Unit Has Competing Risks: Cause 1 vs Cause 2   │
       └───────────────────────────┬────────────────────────────┘
                                   │
              Joint Improved Adaptive Progressive Censoring
                 (Adaptive, Balanced, & Stopping Rules)
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│     Classical Inference (MLE)   │       │       Bayesian Inference        │
│ • Likelihood splits by pop      │       │ • Informative & Weakly Info     │
│ • Closed-form shape MLEs        │       │ • Collapsed Metropolis-Gibbs    │
│ • Scale existence test (Ψ > 0)  │       │ • Exact incomplete beta factor  │
│ • Order-restricted MLEs (pool)  │       │ • SELF, LINEX, & GELF losses    │
│ • Observed Fisher ACIs / delta  │       │ • 95% HPD credible intervals    │
└─────────────────────────────────┘       └─────────────────────────────────┘
                                   │
         ┌─────────────────────────┴─────────────────────────┐
         ▼                                                   ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│ 22 Simulation Scenarios (N=1000)│       │      Real Data Applications     │
│ • Designs D1, D2, D3            │       │ • Boeing 720 Aircraft (Proschan)│
│ • Patterns: Early, Mid, Late    │       │ • Electrical Appliance (Lawless)│
│ • Thresholds: PA, PB            │       │ • Goodness-of-fit & LRT tests   │
└─────────────────────────────────┘       └─────────────────────────────────┘
```

**Key Findings:**
1. **Likelihood Factorization:** The joint likelihood factorizes cleanly by population ($\ell = \ell_1 + \ell_2$) because latent causes share the population-specific scale parameter $\lambda_d$.
2. **Existence Condition:** The finite Lomax scale MLE exists if and only if $\Psi_d = \frac{N_d Q_d}{2 W_d} - S_{1d} > 0$. When $\Psi_d \le 0$, the likelihood is monotonically decreasing and is maximized in the exponential boundary limit ($\lambda_d \to 0$, $\alpha_{dr} \to \infty$ with $\alpha_{dr}\lambda_d = \text{const}$).
3. **Instability of MLEs:** In finite samples, the MLE exists in only $51\% - 79\%$ of replications and exhibits severe volatility.
4. **Superiority of Bayesian Inference:** Bayesian estimates under proper priors (even weakly informative $\text{Gamma}(1,1)$) always exist and dramatically reduce mean squared error (MSE) across all designs.
5. **Exact Order-Restricted Inference:** When $\alpha_{d1} \le \alpha_{d2}$, the MLE pools shape parameters in closed form when the unconstrained estimates violate the constraint. The Bayesian posterior conditional density integrates analytically into a regularized incomplete beta function, enabling exact inverse-CDF draws.

---

## 📁 Project Repository Architecture

```
Lomax-JIAPCS-Competing-Risks/
│
├── main.pdf                          # Primary 14-page verified manuscript (0 broken refs)
├── Lomax_JIAPCS_Project_Report.pdf   # Standalone copy for direct browser reading on GitHub
├── Lomax_JIAPCS_Presentation.pptx   # Complete 15-slide editable PowerPoint presentation
├── main.tex                          # Complete LaTeX source adhering strictly to template
├── generic.sty                       # Preserved template style file (\usepackage{generic})
├── reference.bib                     # BibTeX database (all 15 cited references)
├── bibliography.bib                  # Companion template bibliography file
├── fig1.png                          # High-res Figure 1 (Simulation summary: existence, MSE, coverage)
├── fig2.png                          # High-res Figure 2 (Boeing 720 aircraft air-conditioning analysis)
├── fig3.png                          # High-res Figure 3 (Nelson electrical appliance competing risks)
├── README.md                         # This repository documentation
│
└── code/                             # Self-contained Python scripts and simulation results
    ├── lomax_joint.py                    # Core estimation library (MLE, profile solver, Fisher Info, MCMC)
    ├── run_sim_joint.py                  # Simulation runner across designs D1, D2, D3 (1000 reps)
    ├── realdata.py                       # Boeing 720 aircraft air-conditioning dataset loader
    ├── real_joint.py                     # Aircraft joint model fitting & profile likelihoods
    ├── appliance.py                      # Nelson (1982) electrical appliance competing risks analysis
    ├── assets.py                         # Plotting routines for high-res figures and tables
    ├── extras_real.py                    # Goodness-of-fit and diagnostic tests
    ├── sim_point_all.csv                 # Point estimation Monte Carlo results (22 scenarios)
    ├── sim_interval_all.csv              # Interval estimation Monte Carlo results (ACI & HPD)
    ├── sim_meta_all.csv                  # Scenario metadata and runtime stats
    ├── appliance.json                    # Processed appliance data
    └── extras_real.json                  # Goodness-of-fit test outputs and p-values
```

---

## 🚀 Quick Start & Reproducibility Guide

### 1. Prerequisites
Ensure Python 3.9+ is installed:
```bash
pip install numpy scipy pandas matplotlib
```

### 2. Run Real-Data Analyses
Navigate into the `code/` directory:
```bash
cd code

# (a) Run Boeing 720 aircraft air-conditioning failure analysis:
python real_joint.py

# (b) Run Nelson electrical appliance competing risks analysis:
python appliance.py

# (c) Run diagnostic and goodness-of-fit tests:
python extras_real.py
```

### 3. Generate High-Resolution Figures and Tables
```bash
python assets.py
```

### 4. Re-run Full Monte Carlo Simulation (Optional)
To regenerate all 22 scenarios (1000 replications each):
```bash
python run_sim_joint.py
```

---

## 📄 LaTeX Manuscript Compilation

To recompile `main.tex` into `main.pdf`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

Or upload all files directly to [Overleaf](https://www.overleaf.com/) and compile using **pdfLaTeX**.

---

## 📜 Academic Notice & Citation

This repository is submitted as a course project for **DA2141: Statistics for Data Science** at the **Mehta Family School of Data Science and Artificial Intelligence, Indian Institute of Technology Guwahati (IIT Guwahati)**. All rights are reserved to the authors.

If citing this work, please use:

```bibtex
@article{krishna_bhargav_2026,
  title={Classical and Bayesian Estimation of the Lomax Distribution under Joint Improved Adaptive Progressive Censored Competing Risks Data},
  author={Krishna, Ruttala Vamsi and Bhargav, Bagadi},
  journal={Course Project Report, DA2141: Statistics for Data Science, IIT Guwahati},
  year={2026}
}
```
