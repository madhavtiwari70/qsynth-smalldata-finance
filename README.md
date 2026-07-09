# 🔬 IQP-FinanceSynth

**Quantum Generative Models for Financial Synthetic Data**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2503.02934-b31b1b.svg)](https://arxiv.org/abs/2503.02934)

> Generate realistic synthetic financial data using Instantaneous Quantum Polynomial (IQP) circuits — trained entirely on classical hardware, deployable on quantum hardware.

---

## Why This Exists

Financial machine learning suffers from a fundamental problem: **small datasets**. Fraud detection, rare event modelling, credit scoring for emerging markets, and stress testing all require data that is either scarce, private, or expensive to collect.

Classical synthetic data generators (GANs, VAEs, RBMs) struggle with small datasets because they need large amounts of data to avoid overfitting. IQP circuits offer a different approach — they are trained on the **MMD loss** using an efficient classical algorithm and can be deployed on quantum hardware for sampling where classical simulation becomes hard. How few samples they actually need to converge turns out to depend heavily on the class — see [Results](#results) below for a concrete, measured breakdown rather than a general claim.

This project implements IQP-based synthetic data generation specifically for financial tabular data, with:
- Binarization pipelines designed for financial features (returns, volatility, volume, credit scores)
- Per-class generation (e.g. fraud vs non-fraud, default vs non-default)
- Sample complexity analysis — find the minimum dataset size you actually need
- MMD² evaluation metrics (mode-collapse detection is on the roadmap, not yet implemented)
- Classical baselines (Bernoulli, RBM) for comparison

---

## Installation

```bash
git clone https://github.com/yourusername/iqp-finance-synth
cd iqp-finance-synth
pip install -e .
```

**IQPopt (required, not on PyPI):**
IQPopt is distributed from source only, so it must be installed separately:
```bash
pip install git+https://github.com/XanaduAI/iqpopt.git
```

**Other dependencies** (installed automatically by `pip install -e .`, listed here for reference):
```bash
pip install pennylane jax jaxlib numpy pandas scikit-learn matplotlib seaborn
```

---

## Quick Start

```python
from iqp_finance import FinancialDataPreprocessor, IQPFinanceGenerator, SyntheticEvaluator

# 1. Load your financial data (CSV with numeric features + binary label)
import pandas as pd
df = pd.read_csv("your_fraud_data.csv")

# 2. Preprocess — binarize financial features
preprocessor = FinancialDataPreprocessor(
    n_qubits=16,          # number of qubits / binary feature dimensions
    binarize_method="quantile",  # "quantile", "threshold", or "zscore"
    n_quantiles=4         # discretize into 4 bins per feature
)
X_binary, feature_map = preprocessor.fit_transform(df.drop("label", axis=1))
y = df["label"].values

# 3. Train IQP generator per class
generator = IQPFinanceGenerator(
    n_qubits=16,
    local_gates=3,        # 3-local gates (560 params) — sweet spot for 16 qubits
    sigma=3.0,            # MMD bandwidth
    lr=0.001,
    n_steps=1000,
)
generator.fit(X_binary, y)

# 4. Generate synthetic samples
X_synthetic, y_synthetic = generator.sample(n_samples=500)

# 5. Evaluate quality
evaluator = SyntheticEvaluator(sigma=3.0)
results = evaluator.evaluate(X_binary, X_synthetic, y, y_synthetic)
evaluator.plot_results(results)
```

---

## Use Cases

| Use Case | Dataset Size | Why IQP Helps |
|---|---|---|
| Fraud detection | 50–500 fraud cases | Augment rare fraud class |
| Credit default | 100–300 defaults | Small emerging market datasets |
| Stress testing | 20–100 crisis events | Historical crises are rare |
| Options pricing | 50–200 exotic trades | Rare instrument transactions |

---

## Project Structure

```
iqp-finance-synth/
├── iqp_finance/
│   ├── __init__.py
│   ├── preprocessor.py      # Financial data binarization
│   ├── generator.py         # IQP circuit training & sampling
│   ├── evaluator.py         # MMD², correlation, TSTR/TRTR statistical tests
│   ├── baselines.py         # Classical comparison models
│   └── sample_complexity.py # Minimum dataset size analysis
├── examples/
│   ├── fraud_detection.py   # Credit card fraud example
│   ├── credit_default.py    # Loan default example
│   └── stress_testing.py    # Market stress scenario example
├── notebooks/
│   └── tutorial.ipynb       # Full walkthrough notebook
├── tests/
│   └── test_pipeline.py
└── README.md
```

---

## How It Works

### 1. Binarization of Financial Features

Financial data is continuous — stock returns, credit scores, transaction amounts. We convert these to binary vectors suitable for IQP circuits:

```
Transaction amount $847  →  quantile bin 3  →  binary [0,1,1,0]
Credit score 712         →  quantile bin 2  →  binary [0,1,0,0]
Return -2.3%             →  below median    →  binary [0]
```

### 2. IQP Circuit Training

Each class (fraud/non-fraud, default/non-default) gets its own IQP circuit trained to minimise MMD² against the real data distribution. Training runs entirely on CPU/GPU via JAX — no quantum hardware needed for training.

### 3. Synthetic Sampling

The trained circuit parameters are used to generate new binary samples, which are then mapped back to realistic financial values via the inverse of the binarization pipeline.

### 4. Quality Evaluation

We evaluate using:
- **MMD²** — distributional distance between real and synthetic (overall and per-class)
- **Correlation matrix comparison** — checks if feature relationships are preserved
- **Downstream task performance (TSTR / TRTR)** — train/test a classifier on synthetic vs real data
- **Class balance** — synthetic class proportions vs real
- **Feature parity** — per-feature bit-density comparison

> Note: mode-collapse detection (e.g. via KGEL) is not yet implemented — `IQPFinanceGenerator` currently only does a basic unique-sample-count heuristic during sampling. Treat this as a known gap, not a delivered feature.

---

## Results

Ran on the demo fraud dataset (`generate_demo_fraud_data`: 500 normal transactions, 50 fraud transactions), 16 qubits, `local_gates=3`, `sigma=3.0`.

### Synthetic data quality (IQP generator)

| Metric | Value | Interpretation |
|---|---|---|
| Overall MMD² | 0.00200 | Lower = more similar distributions |
| MMD² — class 0 (normal) | ≈0 (raw unbiased estimate slightly negative, ~−0.00017) | Estimator noise around a well-fit distribution |
| MMD² — class 1 (fraud) | 0.1228 | Meaningfully worse fit than class 0 — see Sample Complexity below |
| Correlation similarity | 0.6449 | 1.0 = identical correlation structure; moderate, not strong, preservation of feature relationships |
| TSTR AUC (train-on-synthetic, test-on-real) | 0.9964 | |
| TRTR AUC (train-on-real, test-on-real) | 1.0 | |

**Caveat on the AUC numbers:** the demo dataset's fraud class is constructed to be trivially separable (fraud transactions are large, late-night, far-from-home, and high-velocity by design — see `generate_demo_fraud_data`). Near-perfect AUC here reflects an easy classification task, not evidence the synthetic data will be similarly informative on a real, harder fraud dataset. Re-run this evaluation against real or less-separable data before treating the AUC gap (or lack thereof) as a meaningful result.

A classical `BernoulliBaseline` comparison is available in the pipeline (`examples/fraud_detection.py`, Step 4) but hasn't been included here yet — run it alongside the IQP results before citing this as evidence the quantum circuit adds value over the simplest possible baseline. *(TODO: fill in Bernoulli MMD²/AUC once run.)*

### Sample complexity — how much training data does the circuit actually need?

`SampleComplexityAnalyzer` was run on `dataset_sizes=[10, 25, 50, 100, 150, 200]`, `n_seeds=3`, `ratio_threshold=1.5`.

**Class 0 (normal, 500 real samples available):**

| n | MMD² |
|---|---|
| 10 | 0.0110 |
| 25 | 0.0074 |
| 50 | 0.0028 |
| 100 | 0.0013 |
| 150 | 0.00060 |
| 200 | 0.00051 |

MMD² decreases steadily and monotonically as training data grows. Minimum sufficient training size (within 1.5× of the n=200 result): **n = 150**.

**Class 1 (fraud, only 50 real samples available):**

| n | MMD² |
|---|---|
| 10 | 0.1523 |
| 25 | 0.1524 |
| 50 | 0.1534 |
| 100+ | not enough real data to test |

MMD² is essentially flat (~0.152) across every size actually tested — it does **not** improve as training data increases from 10 to 50. This is a different, more specific finding than "the model needs more data": within the range we *can* test, more data isn't helping, which points toward a fitting/capacity limitation (gate depth, training steps, learning rate) rather than a pure data-scarcity problem. We can't yet say whether more real fraud data would help, because we only have 50 real fraud samples to test with — which is itself the scarcity problem this project exists to address.

**Methodology note:** "minimum sufficient n" is computed as a ratio against the MMD² at the largest swept size (`n_max`, here 200). A class with fewer real samples than `n_max` (like fraud, at 50) can never reach that normalization point, so `minimum_n` comes back as `None` for it by construction — this is a limitation of the current ratio-based method, not evidence the class never converges. Interpreting a class's row in this table requires checking it actually has data out to `n_max` first.

---

## Citation

This project builds on:

```bibtex
@article{recio2025train,
  title={Train on classical, deploy on quantum: scaling generative quantum machine learning to a thousand qubits},
  author={Recio-Armengol, Erik and Ahmed, Shahnawaz and Bowles, Joseph},
  journal={arXiv preprint arXiv:2503.02934},
  year={2025}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contributing

Pull requests welcome. Please open an issue first to discuss proposed changes.
