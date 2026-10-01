<div align="center">

# Dual-Agent ER-TRPO for House Price Prediction

### Semi-supervised sample selection, explainable latent-feature refinement, and Homotopy-guided multi-fidelity hyperparameter optimization

**ER-TRPO · Semi-Supervised Learning · LSTM–MLP Prediction · LIME · K-Means Uncertainty · Homotopy Optimization**

</div>

## Overview

This repository implements the house-price prediction framework associated with the manuscript:

> **A TRPO-based multi-agent combining semi-supervised learning and explainable feature selection for house price prediction**

The framework addresses house-price regression under partially labeled training data by combining two reinforcement-learning agents with an LSTM–MLP prediction model.

The implementation contains four main mechanisms:

1. **Agent 1 — semi-supervised sample selection**  
   Determines whether an unlabeled training sample should receive the current model prediction as a continuous pseudo-target. Reliability is assessed using entropy derived from soft K-means cluster memberships in the latent representation space.

2. **Agent 2 — feature-selection and prediction policy**  
   Operates over LSTM latent representations and MLP relevance scores. It sequentially decides whether to select another latent feature or terminate the interaction with a house-price prediction action. LIME-derived local importance scores provide feature-selection feedback.

3. **Entropy-Regularized Trust-Region Policy Optimization (ER-TRPO)**  
   Both agents use separate categorical policies and value networks. Policy updates combine a TRPO-style surrogate objective, an entropy contribution, a KL-divergence trust region, conjugate-gradient optimization, and backtracking line search.

4. **Homotopy-guided multi-fidelity hyperparameter optimization**  
   Hyperparameter search proceeds through five Homotopy stages and progressively increasing training budgets. Candidate generation combines random sampling with a KDE-based model-guided search over configurations observed during previous evaluations.

The implementation is designed for the **HouseTS** and **UK House Price (UKHP)** experiments described in the manuscript, while accepting generic tabular representations of these datasets through the command line.

The executable pipeline performs preprocessing, random train/validation/test splitting, labeled/unlabeled partitioning, hyperparameter optimization, final training, test evaluation, repeated runs, model saving, and aggregate metric reporting.

---

## Key Features

### Dual-agent learning

The repository uses two independent ER-TRPO agents rather than one policy for all learning decisions.

- **Agent 1** controls pseudo-label acceptance.
- **Agent 2** controls sequential latent-feature selection and prediction termination.
- The agents maintain separate policy and value networks.
- Agent 1 affects the prediction model indirectly by changing the composition of the labeled training set.
- Updated prediction-model representations subsequently change the information observed by Agent 1.

This implements the asymmetric feedback structure described in the manuscript without sharing policy parameters directly between the two agents.

### Semi-supervised learning

The initial training split is divided into:

- a labeled subset, and
- an unlabeled subset.

The labeled fraction is configurable with `--labeled-fraction` and defaults to `0.30`.

For every unlabeled sample, Agent 1 observes:

```text
[LSTM latent representation, current predicted house price]
```

Its action space is:

```text
0 -> reject the pseudo-label for the current epoch
1 -> attempt to accept the sample
```

Accepted samples receive the current house-price prediction as a continuous pseudo-target and are moved into the labeled training subset.

Rejected samples remain available for later SSL epochs.

### Cluster-entropy uncertainty estimation

The currently labeled samples are encoded by the LSTM and clustered with K-means.

For an unlabeled latent representation \(z\), distances to the cluster centers are transformed into a soft membership distribution:

\[
p_k =
\frac{\exp(-d_k)}
{\sum_j \exp(-d_j)}
\]

where \(d_k\) is the Euclidean distance to cluster center \(k\).

The uncertainty score is then:

\[
H(z) = -\sum_k p_k \log(p_k)
\]

Lower entropy represents a more concentrated association with the labeled-data structure.

The acceptance threshold follows exponential decay:

\[
\delta_t = \delta_0 \exp(-\mu t)
\]

where:

- \(\delta_0\) is `delta_init`,
- \(\mu\) controls threshold decay,
- \(t\) is the current SSL epoch.

In the implementation, Agent 1 receives:

| Condition | Reward |
|---|---:|
| Select and \(H(z)\leq\delta_t\) | `+1` |
| Select and \(H(z)>\delta_t\) | `-1` |
| Reject | `0` |

### LSTM–MLP prediction model

`PriceFeatureModel` consists of:

1. a stacked LSTM,
2. a configurable MLP backbone,
3. a scalar price-prediction head,
4. a latent-feature relevance head.

For each input sequence, the LSTM returns the final temporal representation.

The MLP then generates:

```text
price prediction
+
one relevance score per latent dimension
```

The internal latent dimensionality is fixed to:

```text
64
```

in the current implementation.

### Explainable latent-feature selection

Agent 2 operates over a state consisting of:

```text
[LSTM latent representation,
 MLP relevance scores,
 active-feature mask]
```

Its two actions are:

```text
0 -> select another latent feature
1 -> perform the prediction action and terminate the episode
```

When action `0` is selected, the implementation activates the highest-ranked currently inactive feature according to the MLP relevance scores.

Local feature importance is calculated using LIME when the `lime` package is available.

The absolute LIME coefficients are normalized by their largest value:

\[
\tilde{I}_j =
\frac{|I_j|}
{\max_k |I_k|}
\]

The implemented feature-selection reward is:

\[
r_{\text{feature}} =
\vartheta \tilde{I}_j
\]

where `vartheta` is optimized by the hyperparameter search.

For the prediction action:

\[
r_{\text{prediction}}
=
-(y-\hat{y})^2
\]

The active-feature mask is applied when calculating Agent 2's prediction-action reward.

The normal final prediction path in `DualAgentTrainer.predict()` uses the complete `PriceFeatureModel` output rather than applying a persistent Agent-2 feature mask to test samples.

### LIME fallback

`lime` is imported conditionally.

If `lime.lime_tabular.LimeTabularExplainer` is available, the repository uses it for local explanation.

If the package is unavailable, `LIMEFeatureScorer` falls back to an internal perturbation-based local surrogate based on:

- Gaussian perturbations,
- distance-dependent kernel weights,
- and weighted Ridge regression.

For reproduction of the manuscript's LIME-based methodology, installation of the `lime` package is recommended.

### ER-TRPO policy updates

Each RL component contains:

- a categorical policy network,
- a value network,
- a replay buffer,
- discounted one-step return estimation,
- advantage normalization,
- entropy regularization,
- KL-divergence control,
- conjugate-gradient search,
- Hessian-vector products,
- and backtracking line search.

The implemented policy objective has the form:

\[
L =
\mathbb{E}
\left[
\frac{\pi_\theta(a|s)}
{\pi_{\text{old}}(a|s)}
A(s,a)
\right]
+
\rho\beta\,
\mathcal{H}(\pi_\theta)
\]

where:

- \(A(s,a)\) is the normalized advantage estimate,
- \(\beta\) is the entropy coefficient,
- \(\rho\) is the Homotopy continuation parameter,
- \(\mathcal{H}\) is categorical policy entropy.

Policy updates are accepted only when they improve the surrogate objective and remain within the KL trust region.

### Homotopy-guided hyperparameter search

The repository implements the hyperparameter optimizer directly in `experiment.py`.

The search proceeds through:

```text
rho = 0.00
rho = 0.25
rho = 0.50
rho = 0.75
rho = 1.00
```

Thus entropy regularization is progressively activated during the Homotopy continuation.

Within every Homotopy stage, candidate configurations are evaluated at increasing epoch budgets.

The default budget progression is:

```text
10 -> 30 -> 90 epochs
```

because:

```text
min_budget = 10
eta        = 3
max_budget = 90
```

After each budget level, only the strongest configurations are promoted.

Search history from earlier Homotopy stages is retained for candidate generation at later stages.

---

## Proposed Architecture

The implemented architecture can be summarized as:

```text
                         ┌─────────────────────────────┐
                         │      Training Dataset       │
                         └──────────────┬──────────────┘
                                        │
                           labeled / unlabeled split
                                        │
                         ┌───────────────┴───────────────┐
                         │                               │
                         ▼                               ▼
                 Labeled samples                Unlabeled samples
                         │                               │
                         ▼                               │
                ┌────────────────┐                      │
                │ LSTM Encoder   │◄─────────────────────┤
                └───────┬────────┘                      │
                        │ latent representation          │
                        ▼                               │
                ┌────────────────┐                      │
                │  MLP Backbone  │                      │
                └───────┬────────┘                      │
                        │                               │
              ┌─────────┴──────────┐                    │
              ▼                    ▼                    │
       Price prediction      Relevance scores           │
              │                    │                    │
              │                    ▼                    │
              │              LIME importance            │
              │                    │                    │
              │                    ▼                    │
              │             ┌─────────────┐             │
              │             │   Agent 2   │             │
              │             │ ER-TRPO     │             │
              │             └──────┬──────┘             │
              │                    │                    │
              │             feature-selection           │
              │             / prediction actions        │
              │                                         │
              └──────────────┐                          │
                             ▼                          │
                       ┌─────────────┐                  │
                       │   Agent 1   │◄─────────────────┘
                       │ ER-TRPO     │
                       └──────┬──────┘
                              │
               cluster entropy + policy action
                              │
                  ┌───────────┴───────────┐
                  │                       │
                accept                  reject
                  │                       │
                  ▼                       │
            pseudo-target                 │
                  │                       │
                  ▼                       │
          expand labeled set              │
                  │                       │
                  └──────────┬────────────┘
                             │
                             ▼
                  retrain prediction model
                             │
                             └──── feedback loop
```

Each candidate architecture and training configuration is evaluated under the Homotopy-guided multi-fidelity search before final test evaluation.

---

## Framework Workflow

A complete execution of `main.py` follows this sequence:

```text
Input table
    │
    ▼
Validate dataset path and format
    │
    ▼
Remove invalid target rows
    │
    ▼
Optional date parsing / temporal feature extraction
    │
    ▼
Random 70/15/15 split
    │
    ├────────── Train
    │
    ├────────── Validation
    │
    └────────── Test
    │
    ▼
Fit IQR bounds on training data
    │
    ▼
Apply training-derived IQR filtering to all splits
    │
    ▼
Fit imputers / one-hot encoder / MinMaxScaler on training data
    │
    ▼
Transform validation and test using training-fitted preprocessing
    │
    ▼
Optional sequence construction
    │
    ▼
For each experimental seed
    │
    ▼
Homotopy stage 0.00
    │
    ├── candidate configuration generation
    ├── 10-epoch evaluation
    ├── successive promotion
    ├── 30-epoch evaluation
    └── 90-epoch evaluation
    │
    ▼
Homotopy stages 0.25, 0.50, 0.75, 1.00
    │
    ▼
Choose lowest-validation-RMSE configuration at rho = 1.0
    │
    ▼
Initialize final dual-agent model
    │
    ▼
Split training data into labeled / unlabeled subsets
    │
    ▼
Initial predictor warm-up
    │
    ▼
Iterative SSL + Agent 2 + predictor training
    │
    ▼
Predict untouched test split
    │
    ▼
RMSE / MAPE / RMSLE / R²
    │
    ▼
Save models, configuration, and per-seed metrics
    │
    ▼
Aggregate mean ± standard deviation across seeds
```

---

## Datasets

The manuscript evaluates the framework using two real-world housing datasets.

| Dataset identifier | Dataset | Study role | Target |
|---|---|---|---|
| `house_ts` | HouseTS | U.S. multimodal/spatiotemporal housing benchmark | Continuous house-price value |
| `ukhp` | UK House Price data | England and Wales transaction-level housing data | Transaction price |

### HouseTS

The manuscript describes HouseTS as a large-scale U.S. housing benchmark covering more than 6,000 ZIP codes across 30 metropolitan areas, with monthly observations combining housing-market, socioeconomic, neighborhood/POI, and temporal information.

The tabular component described in the paper contains housing-market, census/socioeconomic, and point-of-interest variables.

### UKHP

The UKHP experiment is based on HM Land Registry Price Paid Data for residential transactions in England and Wales.

The manuscript describes variables including temporal, spatial, and property-related information such as:

- transaction date,
- postcode/location information,
- property type,
- new-build status,
- tenure,
- and transaction price.

### Dataset acquisition

The repository does **not** contain:

- dataset download scripts,
- bundled copies of HouseTS,
- bundled copies of UKHP,
- or automatic retrieval from external sources.

Datasets must therefore be obtained separately and supplied to `main.py` with `--data-path`.

### Dataset identifier behavior

The required `--dataset` argument accepts:

```text
house_ts
ukhp
```

In the current implementation, this argument identifies the experiment but does not activate a dataset-specific parser or fixed schema.

The actual input table is processed generically using the column names supplied through the command-line options.

---

## Input File Formats

`HousingDataProcessor.load_table()` accepts:

| Extension | Reader |
|---|---|
| `.csv` | `pandas.read_csv()` |
| `.txt` | `pandas.read_csv()` |
| `.parquet` | `pandas.read_parquet()` |
| `.pq` | `pandas.read_parquet()` |
| `.xlsx` | `pandas.read_excel()` |
| `.xls` | `pandas.read_excel()` |

The target column must be explicitly supplied with:

```text
--target-column
```

The input file must be a tabular dataset in which each row represents one observation.

For Parquet or Excel input, the corresponding pandas-compatible storage/reader backend must also be available in the Python environment.

---

## Data Preparation and Preprocessing

Preprocessing is implemented in `data_pipeline.py`.

### 1. Target validation

The configured target column must exist.

Target values are converted to numeric form using:

```python
pd.to_numeric(..., errors="coerce")
```

Rows with non-numeric, missing, or non-finite target values are removed.

### 2. Optional date processing

When `--date-column` is supplied and that column exists, it is converted with `pandas.to_datetime()`.

Rows with invalid parsed dates are removed.

The following derived variables are created:

```text
__year
__month
__quarter
__time_index
```

`__time_index` is used internally during date handling but is removed before feature transformation.

The original date column is also excluded from the model input.

### 3. Random dataset partitioning

The cleaned dataset is divided using `train_test_split()`:

```text
70% training
15% validation
15% test
```

The split uses:

```python
shuffle=True
random_state=seed
```

Therefore, the current repository uses **random splitting**, including when a date column is supplied.

It does not implement chronological splitting.

### 4. IQR outlier filtering

Interquartile-range limits are learned exclusively from the training split.

For each numeric feature and the target:

\[
IQR = Q_3-Q_1
\]

with limits:

\[
[Q_1-1.5IQR,\;Q_3+1.5IQR]
\]

The resulting training-derived bounds are then applied to:

- training,
- validation,
- and test data.

Missing feature values are not removed by this filtering step because they are handled later by imputation.

### 5. Numerical preprocessing

Numeric and Boolean feature columns are processed by:

```text
Median imputation
      ↓
MinMaxScaler
```

The scaler is fitted on the training split only.

### 6. Categorical preprocessing

Categorical columns are processed using:

```text
constant missing-value imputation
      ↓
missingness indicator
      ↓
one-hot encoding
```

The missing-category placeholder is:

```text
__MISSING__
```

One-hot encoding uses:

```python
handle_unknown="ignore"
```

allowing categories not observed during training to be ignored safely when transforming validation or test data.

### 7. Target scaling

The current implementation does **not** apply the feature `MinMaxScaler` to the target.

Target values remain in their input numeric scale after cleaning and IQR filtering.

### 8. Sequence construction

The default:

```text
--sequence-length 1
```

converts each observation to a one-step sequence:

```text
[n_samples, 1, n_features]
```

For a sequence length \(L>1\), sliding windows are created independently inside each already-created dataset split.

The target associated with each window is the target of its final row.

No explicit chronological sorting or group-wise sequence construction is performed before window creation.

---

## Prevention of Data Leakage

The implementation contains several explicit train/validation/test separation mechanisms.

### Training-fitted preprocessing

The following are fitted using **training data only**:

- IQR outlier boundaries,
- numerical median imputation,
- numerical MinMax scaling,
- categorical missing-value processing,
- one-hot encoding.

Validation and test data are transformed using those fitted training transformations.

### Hyperparameter optimization

Candidate configurations are trained using the training split.

The HPO objective is:

```text
validation RMSE
```

The test set is not used by `evaluate_config()` when choosing hyperparameters.

### Final evaluation

After HPO, a fresh `DualAgentTrainer` is created from the selected configuration.

The final trainer is fitted using the training split and then evaluated on the held-out test split.

### Search subsampling

When `--max-search-rows` limits the number of training rows used during HPO, only the **training split** is subsampled.

The full validation split remains the source of the configuration-selection RMSE.

---

## Repository Structure

```text
repository/
│
├── main.py
├── data_pipeline.py
├── models.py
├── er_trpo.py
├── experiment.py
└── README.md
```

### `main.py`

Primary executable entry point.

Responsibilities include:

- command-line argument parsing,
- dataset loading,
- repeated-seed experiment orchestration,
- preprocessing invocation,
- Homotopy-guided HPO,
- final model training,
- test evaluation,
- model checkpoint saving,
- configuration saving,
- per-seed metric saving,
- and aggregate summary generation.

### `data_pipeline.py`

Contains dataset and preprocessing utilities:

- `DatasetSplit`,
- `HousingDataProcessor`,
- `split_labeled_unlabeled()`.

It handles:

- input table loading,
- cleaning,
- date processing,
- random dataset splitting,
- IQR filtering,
- imputation,
- one-hot encoding,
- MinMax scaling,
- sequence construction,
- and initial labeled/unlabeled partitioning.

### `models.py`

Defines neural-network components:

- `PriceFeatureModel`,
- `PolicyNetwork`,
- `ValueNetwork`,
- activation-function mapping.

`PriceFeatureModel` contains the LSTM encoder, MLP backbone, price head, and feature-relevance head.

### `er_trpo.py`

Implements the RL optimization machinery:

- `Transition`,
- `ReplayBuffer`,
- flat parameter manipulation,
- flattened autograd,
- conjugate gradient,
- `ERTRPOAgent`,
- value-function training,
- entropy-regularized surrogate objective,
- Fisher/Hessian-vector products,
- KL trust-region constraint,
- and backtracking line search.

### `experiment.py`

Contains the main scientific workflow:

- configuration dataclasses,
- reproducibility utilities,
- regression metrics,
- `LIMEFeatureScorer`,
- `DualAgentTrainer`,
- Agent 1 SSL interaction,
- Agent 2 feature-selection interaction,
- K-means uncertainty estimation,
- hyperparameter search space,
- Homotopy-guided multi-fidelity search,
- validation objective evaluation,
- final training,
- and test evaluation.

### Internal dependencies

The high-level import structure is:

```text
main.py
├── data_pipeline.py
└── experiment.py
    ├── data_pipeline.py
    ├── er_trpo.py
    │   └── models.py
    └── models.py
```

---

## Requirements

### Reference software environment

The associated manuscript reports the following experimental environment:

- Python 3.10
- PyTorch 2.0
- CUDA 11.8
- NumPy 1.24
- Pandas 2.0
- scikit-learn 1.3
- LIME 0.2.0.1

The current source code directly imports:

- `numpy`
- `pandas`
- `torch`
- `scikit-learn`

and conditionally imports:

- `lime`

The hyperparameter search contained in this repository is implemented directly in `experiment.py`; it does not import HpBandSter.

### Core Python packages

A practical environment requires:

```text
numpy
pandas
torch
scikit-learn
lime
```

Additional pandas backends may be necessary when using Excel or Parquet input rather than CSV.

---

## Environment Setup

### Create a virtual environment

```bash
python -m venv venv
```

### Activate on Linux/macOS

```bash
source venv/bin/activate
```

### Activate on Windows

```bash
venv\Scripts\activate
```

### Install the core dependencies

To reproduce the LIME-enabled implementation:

```bash
pip install numpy pandas torch scikit-learn lime
```

If exact manuscript-era versions are required for a reproduction environment, use the versions listed in the reference software environment above, subject to compatibility with the local Python/CUDA installation.

---

## Main Implementation File

The primary executable file is:

```text
main.py
```

Its `main()` function performs the complete repository workflow:

1. parses command-line options,
2. creates the output directory,
3. chooses one or all configured random seeds,
4. loads the dataset,
5. preprocesses and partitions the data,
6. defines the validation objective,
7. runs Homotopy-guided multi-fidelity hyperparameter optimization,
8. trains the final dual-agent framework,
9. evaluates the test split,
10. saves model parameters,
11. saves the optimized configuration,
12. saves per-run metrics,
13. aggregates metrics across runs,
14. writes `summary.json`.

---

## Quick Start

With a CSV dataset and a known target column:

```bash
python main.py \
    --data-path /path/to/housing.csv \
    --dataset house_ts \
    --target-column price
```

For Windows PowerShell, the same command may be written on one line:

```bash
python main.py --data-path C:\path\to\housing.csv --dataset house_ts --target-column price
```

By default, this runs the complete experiment using five seeds:

```text
42
123
2024
3407
5189
```

and writes results under:

```text
outputs/
```

For a faster single-seed run:

```bash
python main.py \
    --data-path /path/to/housing.csv \
    --dataset house_ts \
    --target-column price \
    --single-seed 42
```

---

## Basic Usage

The minimum required syntax is:

```bash
python main.py \
    --data-path PATH \
    --dataset {house_ts,ukhp} \
    --target-column TARGET_COLUMN
```

Display the parser help with:

```bash
python main.py --help
```

---

## Dataset-Specific Execution Examples

### HouseTS

Example:

```bash
python main.py \
    --data-path ./data/house_ts.csv \
    --dataset house_ts \
    --target-column price \
    --date-column date
```

Replace `price` and `date` with the actual corresponding column names in the local HouseTS table.

### UKHP

Example:

```bash
python main.py \
    --data-path ./data/ukhp.csv \
    --dataset ukhp \
    --target-column price \
    --date-column date
```

Again, the repository does not impose fixed target or date column names. The options must match the columns present in the supplied table.

### Sequence input

For a sequence length of 12:

```bash
python main.py \
    --data-path ./data/house_ts.csv \
    --dataset house_ts \
    --target-column price \
    --date-column date \
    --sequence-length 12
```

Sequence windows are constructed after random train/validation/test splitting.

---

## Command-Line Options

| Argument | Required | Default | Description |
|---|---:|---:|---|
| `--data-path` | Yes | — | Path to a CSV, TXT, Parquet, PQ, XLSX, or XLS input table. |
| `--dataset` | Yes | — | Experiment identifier. Choices: `house_ts`, `ukhp`. |
| `--target-column` | Yes | — | Name of the continuous target column. |
| `--date-column` | No | `None` | Optional date column used to derive year, month, and quarter features. |
| `--group-column` | No | `None` | Accepted by the CLI and stored in `HousingDataProcessor`; the current preprocessing/sequence implementation does not use it for grouping. |
| `--sequence-length` | No | `1` | Number of rows per LSTM input sequence. |
| `--labeled-fraction` | No | `0.30` | Fraction of training observations initially treated as labeled. Must be strictly between 0 and 1. |
| `--device` | No | `cuda` when CUDA is available, otherwise `cpu` | PyTorch device string used for the model and RL agents. |
| `--output-dir` | No | `outputs` | Directory for saved checkpoints and JSON results. |
| `--search-initial-configs` | No | `9` | Number of newly generated candidate configurations per Homotopy stage. |
| `--max-search-rows` | No | `50000` | Maximum number of training rows used per HPO candidate evaluation. Values `<=0` disable this training-row limit. |
| `--single-seed` | No | `None` | Runs one requested seed rather than all five manuscript seeds. |

---

## Configuration Behavior

### `--dataset`

The parser restricts this option to:

```text
house_ts
ukhp
```

The implementation does not use the selected identifier to change preprocessing logic or infer target columns automatically.

### `--group-column`

This option is accepted by `main.py` and passed to `HousingDataProcessor`.

In the current implementation, it is not subsequently used to:

- partition data,
- sort observations,
- create grouped sequences,
- or alter preprocessing.

### `--device`

If CUDA is detected, the default device is:

```text
cuda
```

otherwise:

```text
cpu
```

A specific device can be requested explicitly:

```bash
python main.py \
    --data-path ./data/housing.csv \
    --dataset house_ts \
    --target-column price \
    --device cpu
```

or, when available:

```bash
python main.py \
    --data-path ./data/housing.csv \
    --dataset house_ts \
    --target-column price \
    --device cuda
```

---

## Hyperparameter Optimization

The repository implements a Homotopy-guided BOHB-style search internally through the `HomotopyBOHB` class.

The implementation combines:

- random configuration sampling,
- model-guided candidate sampling,
- kernel-density estimation,
- low/high-performing configuration separation,
- multi-fidelity epoch budgets,
- successive promotion,
- and warm transfer of promising configurations across Homotopy stages.

### Optimization objective

Every candidate is evaluated using:

\[
\text{Validation Objective}
=
RMSE(y_{\mathrm{val}},\hat y_{\mathrm{val}})
\]

Lower RMSE is preferred.

### Homotopy schedule

The continuation sequence is fixed in the implementation:

| Stage | \(\rho\) |
|---:|---:|
| 1 | 0.00 |
| 2 | 0.25 |
| 3 | 0.50 |
| 4 | 0.75 |
| 5 | 1.00 |

The entropy term in ER-TRPO is multiplied by `rho`, so the final stage uses the complete entropy contribution.

### Multi-fidelity budgets

Default optimizer parameters are:

```text
min_budget = 10
max_budget = 90
eta        = 3
```

which creates the progression:

```text
10 -> 30 -> 90
```

Training epochs act as the fidelity resource.

After each budget level:

```python
keep = max(1, len(scored) // eta)
```

Only the best candidates continue to the next budget.

### Candidate generation

Before sufficient search history exists, configurations are sampled randomly.

Once at least eight evaluations are available:

1. historical configurations are sorted by validation loss,
2. approximately the best 15% are treated as the good set,
3. the remaining configurations form the bad set,
4. two Gaussian kernel-density estimators are fitted,
5. 64 newly sampled configurations are scored,
6. the configuration maximizing the difference between good-set and bad-set KDE scores is selected.

During new candidate generation, model-based sampling is used with probability approximately \(2/3\) once history exists; otherwise a fresh random configuration is drawn.

### Homotopy warm history

At the end of each stage, up to five strong configurations become the warm candidate set.

At the next Homotopy stage, up to:

```text
search_initial_configs // 3
```

of these warm configurations are added to newly generated candidates.

The implementation transfers configuration/search history between stages, not neural-network weights.

### Final configuration

After all five Homotopy stages, the selected configuration is the configuration with the lowest recorded validation loss among evaluations performed at:

```text
rho = 1.0
```

---

## Hyperparameter Search Space

### Agent-specific parameters

Both Agent 1 and Agent 2 independently optimize:

| Hyperparameter | Search space | Sampling |
|---|---|---|
| Batch size | `[8, 256]` | Integer uniform |
| Epoch limit | `[32, 1024]` | Integer uniform |
| Learning rate | `[1e-4, 5e-2]` | Log-uniform |
| Activation | `Tanh`, `Leaky ReLU`, `Linear`, `ReLU`, `Sigmoid` | Categorical |
| Dropout | `[0, 1]` | Uniform |
| Entropy coefficient `beta` | `[1e-4, 1e-1]` | Log-uniform |

### Shared/method-specific parameters

| Hyperparameter | Search space | Role |
|---|---|---|
| `delta_init` | `[0.1, 1.0]` | Initial Agent-1 entropy threshold |
| `mu` | `[1e-4, 1e-1]` | Exponential entropy-threshold decay |
| `vartheta` | `[0, 1]` | Scale factor for LIME-based Agent-2 feature reward |
| LSTM layers | `[1, 8]` | Depth of the predictor LSTM |
| MLP layers | `[1, 8]` | Depth of the predictor MLP backbone |

The numerical best values reported by the manuscript are **not** hard-coded into the executable training pipeline.

They are outcomes to be rediscovered through optimization.

---

## Fixed Implementation Settings

The following important values are fixed by the current source code rather than included in the HPO search:

| Setting | Value |
|---|---:|
| Predictor latent dimension | `64` |
| Default K-means maximum clusters | `8` |
| Replay-buffer capacity | `100000` |
| ER-TRPO discount factor | `0.99` |
| ER-TRPO maximum KL divergence | `0.01` |
| Fisher/Hessian damping | `0.1` |
| Conjugate-gradient iterations | `10` |
| Value-network optimization steps per policy update | `5` |
| Predictor gradient clipping | `5.0` |
| LIME samples per Agent-2 instance | `128` |
| Maximum LIME background observations | `512` |
| Prediction inference batch size | `2048` |
| Final HPO maximum budget | `90` epochs |

---

## ER-TRPO Implementation

Each `ERTRPOAgent` contains:

```text
PolicyNetwork
ValueNetwork
Adam optimizer for the value network
```

### Policy network

The policy network uses:

```text
Linear
Tanh
Dropout
Linear
Tanh
Linear
```

and outputs categorical action logits.

### Value network

The value model uses:

```text
Linear
Tanh
Linear
Tanh
Linear(1)
```

### Return and advantage estimation

For a sampled transition:

\[
R_t =
r_t +
\gamma(1-d_t)V(s_{t+1})
\]

and:

\[
A_t = R_t - V(s_t)
\]

Advantages are standardized before the policy update.

### Value-function optimization

For every ER-TRPO update, the value network receives five Adam optimization steps minimizing:

\[
\mathcal{L}_V =
\frac{1}{N}
\sum_t
(V(s_t)-R_t)^2
\]

### Policy surrogate

The policy objective contains:

- likelihood-ratio weighted advantage,
- entropy regularization,
- Homotopy scaling.

### Trust-region optimization

The policy gradient is combined with an approximate natural-gradient direction computed using conjugate gradient.

The update scale is determined from the maximum KL constraint.

Backtracking candidate fractions are:

```text
1.0
0.5
0.25
0.125
0.0625
0.03125
```

A candidate step is accepted only when:

1. the surrogate objective improves, and
2. mean KL divergence remains within `max_kl`.

Otherwise, the old policy parameters are restored.

---

## Agent 1: Semi-Supervised Sample Selection

### State

Agent 1 state dimensionality is:

```text
64 latent dimensions + 1 predicted price = 65
```

### Actions

```text
0 -> reject
1 -> select
```

### Initial labeled/unlabeled split

`split_labeled_unlabeled()` randomly permutes the training data.

With the default:

```text
--labeled-fraction 0.30
```

approximately 30% of training samples form the initial labeled subset.

At least two samples are placed in the labeled set.

### K-means clustering

Before processing an unlabeled epoch, current labeled latent representations are clustered.

The actual cluster count is:

```python
min(
    8,
    max(2, len(latent) // 4),
    len(latent)
)
```

K-means uses:

```text
random_state = experimental seed
n_init       = 10
```

### Pseudo-label acceptance

An unlabeled sample is added to the labeled set only when:

```text
Agent 1 chooses action 1
AND
cluster entropy <= current threshold
```

The pseudo-target is the current prediction of `PriceFeatureModel`.

If accepted samples exist, the predictor is immediately retrained for one epoch using the expanded labeled dataset.

Agent 1 is then updated from a randomly sampled replay-buffer mini-batch.

---

## Agent 2: Feature Selection and HPP

### State

Agent 2 concatenates three 64-dimensional components:

```text
latent representation
+
MLP relevance vector
+
active-feature mask
```

giving:

```text
state_dim = 192
```

### Actions

The implementation uses:

```text
0 -> feature-selection action
1 -> prediction action
```

### Feature-selection action

When Agent 2 chooses action `0`:

1. relevance scores of already active features are suppressed,
2. the remaining feature with the largest MLP relevance value is selected,
3. that latent dimension becomes active,
4. its normalized LIME score is used for reward calculation.

### Prediction action

When Agent 2 chooses action `1`:

1. active latent dimensions are preserved,
2. inactive dimensions are masked,
3. the masked latent vector is passed through the predictor's MLP and price head,
4. reward is negative squared prediction error,
5. the interaction terminates.

If no latent feature has been selected before the prediction action, an all-one mask is used, meaning the full latent representation is evaluated.

### Number of Agent-2 interactions per epoch

Agent 2 does not necessarily process the entire labeled set.

The number of labeled instances sampled for RL interaction is:

```python
min(
    len(labeled_data),
    max(32, agent2_batch_size * 2)
)
```

---

## Training Procedure

`DualAgentTrainer.fit()` implements the main training loop.

### 1. Initial labeled/unlabeled split

Only the training partition is passed to the SSL split.

### 2. Predictor warm-up

Before RL interaction begins, the predictor is trained on the initial labeled subset.

Warm-up epochs are:

```python
max(1, min(5, budget_epochs // 5))
```

### 3. Training-length calculation

The number of joint-training epochs is:

```python
min(
    budget_epochs,
    agent1_configured_epochs,
    agent2_configured_epochs
)
```

### 4. Agent 1 SSL epoch

When unlabeled observations remain:

- fit K-means on current labeled latent features,
- update the entropy threshold,
- process unlabeled observations,
- generate policy actions,
- calculate entropy,
- assign rewards,
- accept reliable pseudo-labels,
- preserve rejected samples,
- update Agent 1.

### 5. Agent 2 epoch

Using current labeled observations:

- compute latent representations,
- construct a LIME background set,
- calculate local feature importance,
- execute feature-selection/prediction trajectories,
- collect transitions,
- update Agent 2.

### 6. Predictor update

After both agent interactions, the prediction model is trained for one additional epoch on the current labeled dataset.

### 7. Iteration

The same procedure repeats until the candidate training budget or one of the configured agent epoch limits is reached.

---

## Final Training and Test Evaluation

Once HPO returns the selected configuration, `train_final_and_evaluate()` creates a fresh trainer with:

```text
rho = 1.0
```

The final training budget is:

```python
min(
    90,
    agent1.epochs,
    agent2.epochs
)
```

The final model is trained only on the training partition.

Predictions are then generated for the untouched test partition.

---

## Evaluation Metrics

The implementation computes four regression metrics from test predictions.

### Root Mean Squared Error

\[
RMSE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
}
\]

### Mean Absolute Percentage Error

\[
MAPE =
100
\cdot
\frac{1}{n}
\sum_{i=1}^{n}
\left|
\frac{y_i-\hat y_i}
{y_i}
\right|
\]

For numerical protection, absolute target values smaller than \(10^{-8}\) are replaced by \(10^{-8}\) in the denominator.

### Root Mean Squared Logarithmic Error

\[
RMSLE =
\sqrt{
\frac{1}{n}
\sum_{i=1}^{n}
[
\log(1+\max(y_i,0))
-
\log(1+\max(\hat y_i,0))
]^2
}
\]

### Coefficient of Determination

\[
R^2 =
1-
\frac{
\sum_i(y_i-\hat y_i)^2
}{
\sum_i(y_i-\bar y)^2
}
\]

The implementation uses `sklearn.metrics.r2_score()` for \(R^2\).

---

## Repeated Runs and Statistical Aggregation

Unless `--single-seed` is specified, `main.py` runs the complete preprocessing, HPO, training, and test-evaluation pipeline independently for five seeds:

```text
42
123
2024
3407
5189
```

For every seed:

- a new dataset split is generated,
- preprocessing is refitted,
- HPO is performed again,
- a new final model is trained,
- test metrics are calculated independently.

The final summary reports:

```text
mean
standard deviation
```

for:

- RMSE,
- MAPE,
- RMSLE,
- \(R^2\).

Standard deviation uses:

```python
np.std(..., ddof=1)
```

when more than one seed is run.

For a single-seed execution, the stored standard deviation is `0.0`.

---

## Reproducibility

`seed_everything()` initializes:

```text
Python random
NumPy
PyTorch CPU RNG
PyTorch CUDA RNGs
```

When CUDA is available, the implementation also sets:

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

The selected seed is additionally reused for:

- train/validation/test splitting,
- labeled/unlabeled permutation,
- K-means,
- LIME,
- hyperparameter-search RNG,
- RL sampling,
- training-data shuffling,
- HPO training-row subsampling.

These controls are intended to reduce run-to-run variation.

Exact bitwise determinism can still depend on the installed PyTorch version, device, CUDA stack, and operations used by the local environment.

---

## Running a Single Reproducibility Seed

For seed 42:

```bash
python main.py \
    --data-path ./data/house_ts.csv \
    --dataset house_ts \
    --target-column price \
    --single-seed 42
```

Equivalent examples can be run for:

```text
123
2024
3407
5189
```

If `--single-seed` is omitted, the repository automatically executes all five.

---

## Output Files

The default output location is:

```text
outputs/
```

After a standard five-seed experiment:

```text
outputs/
│
├── seed_42/
│   ├── price_feature_model.pt
│   ├── agent1_policy.pt
│   ├── agent2_policy.pt
│   ├── best_config.json
│   └── metrics.json
│
├── seed_123/
│   ├── price_feature_model.pt
│   ├── agent1_policy.pt
│   ├── agent2_policy.pt
│   ├── best_config.json
│   └── metrics.json
│
├── seed_2024/
│   └── ...
│
├── seed_3407/
│   └── ...
│
├── seed_5189/
│   └── ...
│
└── summary.json
```

### `price_feature_model.pt`

PyTorch `state_dict` for the final `PriceFeatureModel`.

Contains parameters for:

- the LSTM encoder,
- MLP backbone,
- price head,
- relevance head.

### `agent1_policy.pt`

Final Agent-1 policy-network `state_dict`.

### `agent2_policy.pt`

Final Agent-2 policy-network `state_dict`.

### `best_config.json`

The selected hyperparameter configuration for that seed.

Its structure contains:

```json
{
  "agent1": {
    "batch_size": "...",
    "epochs": "...",
    "learning_rate": "...",
    "activation": "...",
    "dropout": "...",
    "beta": "..."
  },
  "agent2": {
    "batch_size": "...",
    "epochs": "...",
    "learning_rate": "...",
    "activation": "...",
    "dropout": "...",
    "beta": "..."
  },
  "delta_init": "...",
  "mu": "...",
  "vartheta": "...",
  "lstm_layers": "...",
  "mlp_layers": "..."
}
```

The actual values are generated by optimization.

### `metrics.json`

Contains the test metrics for one seed:

```json
{
  "rmse": "...",
  "mape": "...",
  "rmsle": "...",
  "r2": "..."
}
```

### `summary.json`

Contains mean and standard deviation across executed seeds:

```json
{
  "rmse": {
    "mean": "...",
    "std": "..."
  },
  "mape": {
    "mean": "...",
    "std": "..."
  },
  "rmsle": {
    "mean": "...",
    "std": "..."
  },
  "r2": {
    "mean": "...",
    "std": "..."
  }
}
```

### Artifacts not saved by the current implementation

The current source does not write:

- preprocessing objects,
- individual test predictions,
- LIME explanations,
- selected-feature histories,
- replay buffers,
- value-network checkpoints,
- optimizer states,
- HPO history,
- training curves,
- visualization files,
- or CSV result tables.

---

## Computational Considerations

The complete experiment can be computationally expensive.

The main cost drivers are:

1. evaluation of many HPO configurations,
2. repeated training at several fidelity budgets,
3. five Homotopy stages,
4. five independent experimental seeds,
5. LSTM training,
6. Agent-1 K-means refitting,
7. local LIME calculations during Agent-2 interaction,
8. conjugate-gradient TRPO updates,
9. repeated final retraining.

### Search-row limit

The default:

```text
--max-search-rows 50000
```

limits only the **training observations used for each HPO candidate evaluation**.

For example:

```bash
python main.py \
    --data-path ./data/house_ts.csv \
    --dataset house_ts \
    --target-column price \
    --max-search-rows 10000
```

reduces HPO training cost but changes the search conditions relative to using the full training set.

To remove the limit:

```bash
--max-search-rows 0
```

### Initial candidate count

Search cost can also be reduced through:

```text
--search-initial-configs
```

For example:

```bash
python main.py \
    --data-path ./data/house_ts.csv \
    --dataset house_ts \
    --target-column price \
    --search-initial-configs 3 \
    --single-seed 42
```

This is useful for code validation or exploratory runs but does not reproduce the default search budget.

### Single-seed runs

Using:

```text
--single-seed
```

avoids repeating the complete pipeline five times and is therefore useful for initial environment verification.

---

## Hardware Acceleration

All neural networks and tensor computations use the device string supplied through:

```text
--device
```

The default is determined at runtime:

```python
"cuda" if torch.cuda.is_available() else "cpu"
```

### CPU

```bash
python main.py \
    --data-path ./data/housing.csv \
    --dataset house_ts \
    --target-column price \
    --device cpu
```

### CUDA

```bash
python main.py \
    --data-path ./data/housing.csv \
    --dataset house_ts \
    --target-column price \
    --device cuda
```

A CUDA request requires a working CUDA-enabled PyTorch environment and compatible hardware.

The manuscript reports experiments performed on an NVIDIA RTX 3090 with 24 GB VRAM.

---

## Methodological Components Implemented

- [x] Tabular dataset loading
- [x] CSV/TXT input
- [x] Parquet input
- [x] Excel input
- [x] Target validation
- [x] Optional date parsing
- [x] Year/month/quarter feature extraction
- [x] 70/15/15 random splitting
- [x] Training-derived IQR filtering
- [x] Median numerical imputation
- [x] Min–max feature scaling
- [x] Categorical missing-value handling
- [x] One-hot categorical encoding
- [x] Optional multi-row sequence construction
- [x] Initial labeled/unlabeled split
- [x] LSTM encoder
- [x] MLP prediction backbone
- [x] Continuous house-price head
- [x] Latent relevance head
- [x] Agent 1 SSL policy
- [x] Agent 2 feature-selection/HPP policy
- [x] Independent value networks
- [x] Replay buffers
- [x] K-means latent clustering
- [x] Soft cluster-membership entropy
- [x] Exponentially decaying pseudo-label threshold
- [x] Continuous pseudo-target assignment
- [x] LIME-based latent importance
- [x] Local Ridge-surrogate fallback
- [x] Sequential active-feature masking inside Agent 2
- [x] Entropy-regularized policy objective
- [x] KL-divergence trust region
- [x] Conjugate-gradient TRPO direction
- [x] Backtracking line search
- [x] Homotopy continuation
- [x] Multi-fidelity epoch budgets
- [x] KDE-guided configuration generation
- [x] Successive configuration promotion
- [x] Validation-RMSE model selection
- [x] Final independent test evaluation
- [x] RMSE
- [x] MAPE
- [x] RMSLE
- [x] \(R^2\)
- [x] Five-seed repeated experiments
- [x] Mean ± standard-deviation aggregation
- [x] PyTorch checkpoint saving
- [x] Best-configuration JSON output
- [x] Per-seed metric JSON output
- [x] Aggregate summary JSON output

---

## Reproducing an Experiment

A representative complete HouseTS execution is:

```bash
python main.py \
    --data-path ./data/house_ts.csv \
    --dataset house_ts \
    --target-column price \
    --date-column date \
    --output-dir outputs
```

After this command, the implementation performs the following steps independently for each of the five default seeds.

1. **Load the table**  
   The file extension determines the pandas reader.

2. **Validate the target**  
   Invalid target observations are removed.

3. **Process the date column**  
   If supplied and available, date values are parsed and temporal fields are extracted.

4. **Create a 70/15/15 random split**  
   Training, validation, and test partitions are generated from the current seed.

5. **Fit IQR boundaries using training data**  
   Training-derived boundaries are subsequently applied to all three partitions.

6. **Fit the preprocessing transformer on training features**  
   Numerical imputation/scaling and categorical encoding parameters are learned without using validation or test rows.

7. **Transform all partitions**

8. **Create LSTM sequence tensors**

9. **Define the HPO validation objective**

10. **Begin the Homotopy search at \(\rho=0\)**

11. **Evaluate candidate hyperparameters at low fidelity**

12. **Promote stronger configurations to larger training budgets**

13. **Carry promising configurations into the next Homotopy stage**

14. **Repeat optimization for \(\rho=0.25, 0.50, 0.75, 1.00\)**

15. **Select the minimum-validation-RMSE configuration from the final Homotopy stage**

16. **Initialize a fresh final dual-agent trainer**

17. **Randomly divide the training partition into labeled and unlabeled subsets**

18. **Warm up the predictor on labeled samples**

19. **Run Agent 1 pseudo-label selection**

20. **Expand the labeled training set with accepted pseudo-targets**

21. **Run Agent 2 LIME-guided feature-selection interactions**

22. **Update both ER-TRPO policies**

23. **Retrain the house-price predictor**

24. **Repeat until the final training budget is exhausted**

25. **Predict the test partition**

26. **Compute RMSE, MAPE, RMSLE, and \(R^2\)**

27. **Save model and policy state dictionaries**

28. **Save the selected configuration**

29. **Save per-seed test metrics**

30. **Repeat the pipeline for the remaining seeds**

31. **Aggregate mean and standard deviation**

32. **Write `summary.json`**

---

## Minimal Smoke-Test Workflow

For initial installation verification, a smaller one-seed search can be invoked with:

```bash
python main.py \
    --data-path ./data/housing.csv \
    --dataset house_ts \
    --target-column price \
    --single-seed 42 \
    --search-initial-configs 3 \
    --max-search-rows 5000
```

This uses valid repository options but reduces the default experimental search effort.

It should be treated as an execution check rather than the complete default experimental protocol.

---

## Troubleshooting

### Dataset path does not exist

The loader raises:

```text
FileNotFoundError: Dataset path does not exist: ...
```

Verify `--data-path`.

### Unsupported file extension

Files outside the implemented CSV/TXT, Parquet/PQ, and Excel formats raise:

```text
ValueError: Unsupported dataset format: ...
```

### Target column is missing

The target name must match the table exactly.

Otherwise preprocessing raises:

```text
ValueError: Target column '...' was not found.
```

### Invalid labeled fraction

`--labeled-fraction` must satisfy:

```text
0 < labeled_fraction < 1
```

### Sequence length is too large

Every split must contain enough rows to create the requested sequence length.

Otherwise:

```text
ValueError: The split contains fewer rows than sequence_length.
```

### LIME is unavailable

The training pipeline does not terminate solely because `lime` cannot be imported.

Instead, the repository uses its built-in perturbation/Ridge surrogate.

Install `lime` when reproducing the manuscript's LIME-based experimental setup.

### CUDA execution fails

Use:

```text
--device cpu
```

if the installed PyTorch package or local hardware does not support CUDA.

### High runtime during HPO

Consider testing the installation first with:

```text
--single-seed
--search-initial-configs
--max-search-rows
```

before launching the complete default search.

---

## Important Implementation Notes

### Paper methodology versus executable behavior

The repository follows the central dual-agent architecture described in the manuscript, but execution documentation should be based on the source code.

In particular:

- the repository implements the hyperparameter search directly rather than importing HpBandSter;
- target values are not MinMax-scaled by `HousingDataProcessor`;
- train/validation/test partitions are random rather than chronological;
- `--group-column` is currently not used to construct grouped sequences;
- Agent 2 feature masks are used during its reinforcement-learning prediction reward, while final `predict()` calls the complete predictor;
- the HPO search history is retained in memory but is not written to disk;
- only the policy networks, not value networks or optimizer states, are saved.

These behaviors are relevant when comparing repository outputs with the methodological description and reported experiments.

---

## Paper

**Title:**  
*A TRPO-based multi-agent combining semi-supervised learning and explainable feature selection for house price prediction*

**Authors:**

- Mahyar Ramezankhani
- Roohallah Alizadehsani
- Siamak Pedrammehr

The manuscript proposes a dual-agent house-price prediction framework in which:

- one ER-TRPO policy performs semi-supervised pseudo-label selection,
- another policy performs explainability-guided feature refinement and prediction decisions,
- LSTM and MLP components provide latent representations and continuous price estimates,
- cluster-membership entropy measures pseudo-label uncertainty,
- LIME provides local feature-relevance information,
- and Homotopy-guided hyperparameter optimization progressively activates the entropy-regularized objective.

This repository provides the executable implementation of that methodological pipeline.

---

## Citation

At the time represented by the supplied manuscript, complete publication metadata such as journal, volume, pages, publication year, and DOI are not provided.

The manuscript can therefore be cited without inventing unavailable publication metadata:

```bibtex
@unpublished{ramezankhani_ertrpo_house_price,
  title  = {A TRPO-based multi-agent combining semi-supervised learning and explainable feature selection for house price prediction},
  author = {Ramezankhani, Mahyar and Alizadehsani, Roohallah and Pedrammehr, Siamak},
  note   = {Manuscript}
}
```

Update the citation after formal publication metadata becomes available.

---

## Contact

Corresponding-author contact information listed in the manuscript:

```text
Ek18128@alumni.bristol.ac.uk
s.pedrammehr@gmail.com
```

Repository-specific implementation issues may also be tracked through the repository's GitHub issue system when hosted publicly.

---

## License

No software license is specified by the supplied source files.

A repository license should be added before public redistribution or reuse terms are asserted.
