# Experiments

This document explains the purpose of each experiment script in this repository. The experiments correspond to the evaluation carried out in Chapter 4 of the TFM *"Differential Privacy in the Registration of Sequential Events"*, which assesses how well the proposed ϵ-differential privacy method (based on Elkoumy et al., extended with macro-temporal perturbation) preserves the utility of sequential event logs after anonymization.

All experiments are run over the three evaluation datasets (`synthetic_data_reg1.csv`, `synthetic_data_reg2.csv`, `synthetic_data_reg3.csv`), typically sweeping the re-identification probability parameter **δ** (e.g. from 0.20 to 0.40).

---

## Experiment 1 — Parameter comparison
**Files: `experiment_1_1.py`, `experiment_1_2.py`**

Evaluates the impact of δ on three aspects of the pipeline:

1. **Filtered cases** — the percentage of cases discarded during the filtering step (a case is dropped when `Pk + δ ≥ θ`). Lower δ (stricter privacy) filters slightly more cases.
2. **Privacy budget (ϵₜ)** — how the average timestamp privacy budget assigned to events changes with δ. Lower δ forces smaller ϵₜ (more noise on timestamps).
3. **Precision of resulting patterns** — the accuracy-loss metric (Equation 4.2 in the thesis) comparing pattern frequencies before/after anonymization, to check that structural utility is preserved.

`experiment_1_1.py` and `experiment_1_2.py` split this analysis (e.g. one script computing the filtering/ϵₜ statistics, the other computing the precision comparison) across the three datasets, producing the plots equivalent to Figures 4.5–4.7 in the thesis.

---

## Experiment 2 — Pattern frequency variation
**Files: `experiment_2_1.py`, `experiment_2_2.py`**

Looks at individual trace patterns rather than aggregate statistics:

1. **Frequency variation per pattern** — for each δ, the most common patterns in the original (filtered) log are compared against their frequency in the anonymized log, expressed as a percentage change. This shows which patterns are most robust (usually the most frequent ones) and which are most distorted by noise (usually rare/minority patterns) — see Table 4.4.
2. **Macro-Temporal Perturbation (MTP) effect** — a comparison between an anonymized log **without** temporal shift (timestamps stay within the original calendar range) and one **with** the shift applied (a random offset of up to *M* months and *D* days per case), verifying that the shift preserves relative event order and inter-event timing while breaking the link to the original calendar period.

`experiment_2_1.py` and `experiment_2_2.py` correspond respectively to these two sub-analyses (pattern-frequency drift, and the with/without MTP comparison).

---
 
## Experiment 3 — Process graph comparison
**File: `experiment_3.py`**
 
Evaluates whether the anonymized logs preserve the underlying process structure by rendering and comparing the process graphs of the original and anonymized event logs. For each dataset, the script builds a graph representation showing nodes (activities) and edges (transitions) annotated with their frequency and relative probability, both for the original log and for the log anonymized with δ = 0.3. This makes it possible to visually verify that the most frequent paths, node proportions, and transition probabilities remain almost identical after anonymization, with deviations typically below 1%.
 
---
 

## Experiment 4 — Machine learning impact (TOTO vs. TPTO)
**Files: `experiment_3_1.py`, `experiment_3_2_toto.py`, `experiment_3_3_tpto.py`**

Measures the **cost of privacy**: whether anonymization degrades the usefulness of the log for a downstream machine learning task — next-activity prediction using a Recurrent Neural Network (RNN).

Two scenarios are compared:

- **TOTO — Train on Original, Test on Original**: the RNN is trained and evaluated on the untouched dataset. This establishes the upper-bound (ideal) performance.
- **TPTO — Train on Privatized, Test on Original**: the RNN is trained on the differentially private log (δ = 0.3, θ = 1, ±2 months/±2 days temporal shift) but evaluated on the original test set, isolating the effect of training on noisy data.

File breakdown:

- `experiment_3_1.py` — shared setup: data preparation, encoding of activity sequences, and RNN model definition used by both scenarios.
- `experiment_3_2_toto.py` — runs the **TOTO** scenario and reports accuracy.
- `experiment_3_3_tpto.py` — runs the **TPTO** scenario (training on the anonymized log produced by the DP pipeline) and reports accuracy for comparison against TOTO.

The difference in accuracy between the two scenarios is the *privacy cost*. In the thesis results (Table 4.7), this difference was **zero** across all three datasets, indicating the anonymization preserves the causal/temporal structure the RNN relies on.
