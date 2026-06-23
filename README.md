# Hybrid Modelling for Metabolic Pathway Optimization: The Impact of Data Availability and Prior Mechanistic Knowledge

## Overview

This repository contains the code, datasets, and experiment workflows for a study on the effect of data abundance and prior knowledge in hybrid metabolic pathway optimization. It compares hybrid mechanistic models and data-driven approaches to predict metabolite dynamics in a p-coumaric acid (pCA) production pathway.

The project is organized around:
- Data generation and noisy dataset creation
- Hybrid kinetic models, neural ODEs, and XGBoost baselines
- Structured experiments for three research questions
- Analysis and visualisation of results

This implementation is part of a Master's thesis, the abstract of the associated thesis: 

> Metabolic engineering offers a promising route toward sustainable production of valuable chemicals by
enabling the design of optimized microbial pathways. However, the vast combinatorial design space
and the high cost of experimental validation make it challenging to efficiently identify optimal strain
designs. Methods such as kinetic approaches that rely on Ordinary Differential Equations (ODEs) are
interpretable, but need detailed knowledge of the underlying pathway. On the other hand, purely data-
driven machine learning methods, require a lot of training data and have more trouble generalising
to unseen conditions. Hybrid modelling approaches, such as Neural Ordinary Differential Equations
(Neural ODEs) and Universal Differential Equations (UDEs), have emerged as a promising solution
by combining mechanistic knowledge with data-driven learning. Despite their potential, it remains un-
clear how much data, and what type of data, is required to obtain reliable predictions in practical, data-
limited settings. This study systematically investigates the relationship between data availability, model
complexity, and predictive performance of hybrid kinetic models for a metabolic pathway producing p-
coumaric acid. Using synthetically generated data, hybrid models with varying levels of mechanistic
knowledge were trained and compared to fully data-driven approaches, including Neural ODEs and
XGBoost. The influence of training set size (number of strains), temporal resolution (number of time-
points), prior mechanistic knowledge, reaction lumping, and observational noise was evaluated across
multiple scenarios. Model performance was assessed in terms of prediction accuracy, training stability,
and generalization to unseen strains. The results demonstrate that increasing the number of strains
is generally more beneficial than increasing the number of timepoints, for predicting final product con-
centration. For predicting full metabolite concentrations it is more beneficial to increase the number of
timepoints, but this requires a more careful trade-off. The hybrid model utilizes the underlying mecha-
nistic knowledge to make moderate predictions even on a very limited number of timepoints. The hybrid
models consistently outperform purely data-driven methods in low-data regimes, particularly when prior
mechanistic knowledge is incorporated. While additional mechanistic knowledge improves predictive
performance and reduces data requirements, it also increases sensitivity to noise and can negatively
affect performance in highly constrained settings. Reaction lumping reduces model complexity but typ-
ically leads to lower performance compared to full network representations, especially when sufficient
temporal data is available. Overall, this study provides practical insights into the trade-offs between
data quantity, data type, and model complexity in hybrid metabolic modelling. The findings highlight
that efficient experimental design should prioritize increasing the diversity of sampled strains over tem-
poral resolution, and that incorporating mechanistic knowledge is most beneficial when data is scarce
but noise levels remain limited.

## Repository Structure

- `dataset/` — synthetically generated dataset of time series data for different strain designs for a p-coumaric acid production pathway, and noisy variants
- `models/` — SBML pathway definitions and lumped reaction models
- `scripts/` — experiment runners, model utilities, dataset creation, and parameter tuning code
- `Experiments/` — generated results, summaries, comparison outputs, and figures
- `Notebooks/` — starting notebook used to load the model and generate the dataset
- `requirements.txt` — Python dependency list

## Key Scripts

### Experiment runners
- `scripts/run_experiments_question_1.py` — evaluate models for question 1 across varying time resolution and strain counts
- `scripts/run_experiments_question_2.py` — investigate model performance under differing structural knowledge masks
- `scripts/run_experiments_question_3.py` — investigate model performance under different levels of reaction lumping
- `scripts/run_experiments_noise.py` — test model robustness on datasets with 5%, 10%, and 20% noise

### Support and utilities
- `scripts/ExperimentRunner.py` — main experiment class supporting HybridModel, NODEModel, and XGBoost
- `scripts/data_generation.py` — load time-series CSV datasets and generate synthetic training data
- `scripts/add_noise_to_dataset.py` — create noisy dataset variants from clean data
- `scripts/create_lumped_models.py` — generate lumped SBML pathway models
- `scripts/Parameter_tuning/` — hyperparameter search and plotting utilities
- `scripts/Analysis/` — result visualization and quantitative comparisons across experiments

## Data Layout

The dataset directories contain CSV files for each metabolite plus strain design matrices:

- `data/` — clean simulation outputs for the pCA model
- `data_noise_5/`, `data_noise_10/`, `data_noise_20/` — noisy versions of the same data with added Gaussian noise

Each dataset folder contains:
- `pCA_model_<metabolite>.csv` — time-series values for all simulated strains
- `pCA_model_strain_designs.csv` — parameter perturbations used to generate each strain

## Models

The repository uses several model formulations:
- `scripts/models/hybrid_model.py` — hybrid mechanistic/data-driven model using the SBML kinetics structure
- `scripts/models/NODE.py` — neural ODE model variant
- `scripts/models/XGBoostModel.py` — data-driven tree ensemble baseline

## Setup

1. Clone this repository:

```bash
git clone https://github.com/Talitha147/hybrid-metabolic-pathway-optimization-data-and-priors.git
cd hybrid-metabolic-pathway-optimization-data-and-priors
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```


## Running Experiments

Use the provided scripts to run experiments from the repository root.

Example: Question 1 experiments

```bash
python scripts/run_experiments_question_1.py
```

Example: Noise experiments

```bash
python scripts/run_experiments_noise.py --noise 10
```

### Notes
- The runners assume the SBML model file path `models/pCA_model_changed_S.xml` exists.
- Experiment results are stored under the `Experiments/` directory.


## Analysis

The repository includes analysis scripts to summarize experiments and generate figures:
- `scripts/Analysis/visualize_results.py`
- `scripts/Analysis/visualize_results_q2.py`
- `scripts/Analysis/visualize_results_q3.py`
- `scripts/Analysis/plot_noise_complexity.py`
- `scripts/Analysis/compare_q2_q3_trajectories.py`

## Notes 

- The SBML model is loaded using
  [`AbeelLab/jaxkineticmodel`](https://github.com/AbeelLab/jaxkineticmodel)
  (van Lent et al., 2025).

