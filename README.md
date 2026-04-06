# 🔬 IQP-FinanceSynth

**Quantum Generative Models for Financial Synthetic Data**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2503.02934-b31b1b.svg)](https://arxiv.org/abs/2503.02934)

> Generate realistic synthetic financial data using Instantaneous Quantum Polynomial (IQP) circuits — trained entirely on classical hardware, deployable on quantum hardware.

---

## Why This Exists

Financial machine learning suffers from a fundamental problem: **small datasets**. Fraud detection, rare event modelling, credit scoring for emerging markets, and stress testing all require data that is either scarce, private, or expensive to collect.

Classical synthetic data generators (GANs, VAEs, RBMs) struggle with small datasets because they need large amounts of data to avoid overfitting. IQP circuits offer a different approach — they are trained on the **MMD loss** using an efficient classical algorithm, require surprisingly few samples to converge, and can be deployed on quantum hardware for sampling where classical simulation becomes hard.

This project implements IQP-based synthetic data generation specifically for financial tabular data, with:
- Binarization pipelines designed for financial features (returns, volatility, volume, credit scores)
- Per-class generation (e.g. fraud vs non-fraud, default vs non-default)
- Sample complexity analysis — find the minimum dataset size you actually need
- MMD² and KGEL evaluation metrics
- Classical baselines (Bernoulli, RBM) for comparison

---

## Installation

```bash
git clone https://github.com/yourusername/iqp-finance-synth
cd iqp-finance-synth
pip install -e .
```

**Dependencies:**
```bash
pip install iqpopt pennylane jax jaxlib numpy pandas scikit-learn matplotlib seaborn
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
    n_qubits=16,          # circuit size (must be perfect square for image layout)
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
│   ├── evaluator.py         # MMD², KGEL, statistical tests
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
- **MMD²** — distributional distance between real and synthetic
- **KGEL** — detects mode collapse and mode imbalance  
- **Correlation matrix comparison** — checks if feature relationships are preserved
- **Downstream task performance** — train/test a classifier on synthetic vs real data

---

## Key Finding: Sample Complexity

A core result of this project is the **minimum dataset size** needed for the IQP circuit to learn a given class distribution. We find that for 16-qubit circuits with 3-local gates:

```
n = 50–100 samples is typically sufficient
n > 200 samples yields diminishing returns
```

This makes IQP circuits particularly attractive for financial applications where labelled data is scarce.

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
