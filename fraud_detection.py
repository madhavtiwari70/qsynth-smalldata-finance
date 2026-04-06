"""
Example: Credit Card Fraud Detection
======================================
Demonstrates IQP synthetic data generation for fraud detection.

The fraud class is heavily imbalanced (typically <1% of transactions).
This example shows how to:
1. Augment the rare fraud class with synthetic data
2. Train a classifier on augmented data
3. Compare against classical baselines

Usage:
    python examples/fraud_detection.py

Or with your own data:
    python examples/fraud_detection.py --data_path your_fraud_data.csv --label_col is_fraud
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from iqp_finance import (
    FinancialDataPreprocessor,
    IQPFinanceGenerator,
    SyntheticEvaluator,
    BernoulliBaseline,
    SampleComplexityAnalyzer,
)


# ================================================
# Synthetic fraud dataset generator (for demo)
# ================================================

def generate_demo_fraud_data(
    n_normal: int = 500,
    n_fraud: int = 50,
    n_features: int = 8,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic fraud dataset for demonstration.

    Normal transactions: low amounts, regular hours, low risk
    Fraud transactions:  high amounts, unusual hours, high risk
    """
    rng = np.random.default_rng(random_seed)

    # Normal transactions
    normal = pd.DataFrame({
        "amount":        rng.exponential(scale=100, size=n_normal),
        "hour_of_day":   rng.integers(8, 20, size=n_normal).astype(float),
        "days_since_last": rng.exponential(scale=3, size=n_normal),
        "merchant_risk": rng.beta(a=2, b=8, size=n_normal),
        "card_age_months": rng.integers(6, 120, size=n_normal).astype(float),
        "n_transactions_24h": rng.integers(0, 5, size=n_normal).astype(float),
        "distance_from_home": rng.exponential(scale=10, size=n_normal),
        "velocity_score": rng.beta(a=3, b=7, size=n_normal),
        "label": 0,
    })

    # Fraud transactions — different distribution
    fraud = pd.DataFrame({
        "amount":        rng.exponential(scale=800, size=n_fraud),
        "hour_of_day":   np.concatenate([
            rng.integers(0, 6, size=n_fraud // 2),
            rng.integers(22, 24, size=n_fraud - n_fraud // 2),
        ]).astype(float),
        "days_since_last": rng.exponential(scale=0.5, size=n_fraud),
        "merchant_risk": rng.beta(a=8, b=2, size=n_fraud),
        "card_age_months": rng.integers(1, 12, size=n_fraud).astype(float),
        "n_transactions_24h": rng.integers(5, 20, size=n_fraud).astype(float),
        "distance_from_home": rng.exponential(scale=200, size=n_fraud),
        "velocity_score": rng.beta(a=7, b=3, size=n_fraud),
        "label": 1,
    })

    df = pd.concat([normal, fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return df


# ================================================
# Main pipeline
# ================================================

def run_fraud_pipeline(
    df: pd.DataFrame,
    label_col: str = "label",
    n_synthetic: int = 200,
    n_qubits: int = 16,
    local_gates_k: int = 3,
    n_steps: int = 1000,
    verbose: bool = True,
):
    """
    Full fraud detection synthetic data pipeline.
    """
    print("\n" + "=" * 60)
    print("  IQP Finance Synth — Fraud Detection Pipeline")
    print("=" * 60)

    # Separate features and labels
    feature_cols = [c for c in df.columns if c != label_col]
    X_raw = df[feature_cols]
    y = df[label_col].values

    print(f"\n  Dataset: {len(df)} transactions")
    print(f"  Features: {feature_cols}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Step 1: Preprocess
    print("\n--- Step 1: Binarizing financial features ---")
    preprocessor = FinancialDataPreprocessor(
        n_qubits=n_qubits,
        binarize_method="quantile",
        n_quantiles=4,  # 2 bits per feature
    )
    X_binary, feature_map = preprocessor.fit_transform(X_raw)
    print(preprocessor.summary())

    # Density diagnostics
    print("\n  Bit densities per feature:")
    densities = preprocessor.get_density_per_feature(X_binary)
    for feat, density in densities.items():
        bar = "█" * int(density * 20)
        print(f"    {feat:25s}: {density:.3f}  {bar}")

    # Step 2: Train IQP generator
    print("\n--- Step 2: Training IQP circuits ---")
    generator = IQPFinanceGenerator(
        n_qubits=n_qubits,
        local_gates=local_gates_k,
        sigma=3.0,
        lr=0.001,
        n_steps=n_steps,
        verbose=verbose,
    )
    generator.fit(X_binary, y)

    # Step 3: Generate synthetic data
    print(f"\n--- Step 3: Generating {n_synthetic} synthetic samples ---")
    X_synth, y_synth = generator.sample(n_samples=n_synthetic)
    print(f"  Synthetic class distribution: {dict(zip(*np.unique(y_synth, return_counts=True)))}")

    # Decode back to financial values
    X_synth_decoded = preprocessor.inverse_transform(X_synth)
    print(f"\n  Sample synthetic transactions (decoded):")
    print(X_synth_decoded.head(3).to_string())

    # Step 4: Train classical baselines
    print("\n--- Step 4: Training classical baselines ---")
    bernoulli = BernoulliBaseline(random_seed=42)
    bernoulli.fit(X_binary, y)
    X_bern, y_bern = bernoulli.sample(n_samples=n_synthetic)
    print("  ✓ Bernoulli baseline trained")

    # Step 5: Evaluate
    print("\n--- Step 5: Evaluating synthetic data quality ---")
    evaluator = SyntheticEvaluator(sigma=3.0)

    results_iqp = evaluator.evaluate(
        X_binary, X_synth, y, y_synth,
        feature_names=list(densities.keys()),
    )
    results_bern = evaluator.evaluate(
        X_binary, X_bern, y, y_bern,
        feature_names=list(densities.keys()),
    )

    print("\n  IQP Generator:")
    evaluator.print_summary(results_iqp)

    print("  Bernoulli Baseline:")
    evaluator.print_summary(results_bern)

    # Comparison
    print("\n--- Comparison: IQP vs Bernoulli ---")
    print(f"  {'Metric':<30} {'IQP':>12} {'Bernoulli':>12} {'Winner':>10}")
    print(f"  {'-'*64}")
    iqp_mmd = results_iqp["mmd2_overall"]
    bern_mmd = results_bern["mmd2_overall"]
    print(f"  {'Overall MMD²':<30} {iqp_mmd:>12.5f} {bern_mmd:>12.5f} "
          f"{'IQP ✓' if iqp_mmd < bern_mmd else 'Bernoulli':>10}")

    iqp_corr = results_iqp["correlation_similarity"]
    bern_corr = results_bern["correlation_similarity"]
    print(f"  {'Correlation Similarity':<30} {iqp_corr:>12.4f} {bern_corr:>12.4f} "
          f"{'IQP ✓' if iqp_corr > bern_corr else 'Bernoulli':>10}")

    iqp_tstr = results_iqp["tstr"].get("tstr_auc", np.nan)
    bern_tstr = results_bern["tstr"].get("tstr_auc", np.nan)
    if not np.isnan(iqp_tstr) and not np.isnan(bern_tstr):
        print(f"  {'TSTR AUC':<30} {iqp_tstr:>12.4f} {bern_tstr:>12.4f} "
              f"{'IQP ✓' if iqp_tstr > bern_tstr else 'Bernoulli':>10}")

    # Step 6: Visualize
    print("\n--- Step 6: Generating evaluation plots ---")
    evaluator.plot_results(
        results_iqp, X_binary, X_synth, y, y_synth,
        feature_names=list(densities.keys()),
        save_path="fraud_evaluation.png",
    )

    return {
        "generator": generator,
        "preprocessor": preprocessor,
        "X_synthetic": X_synth,
        "y_synthetic": y_synth,
        "results_iqp": results_iqp,
        "results_bernoulli": results_bern,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IQP Fraud Detection Example")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to CSV file (uses demo data if not provided)")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--n_synthetic", type=int, default=200)
    parser.add_argument("--n_qubits", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=500)
    args = parser.parse_args()

    if args.data_path:
        df = pd.read_csv(args.data_path)
        print(f"Loaded dataset from {args.data_path}: {df.shape}")
    else:
        print("No data path provided — using generated demo fraud dataset.")
        df = generate_demo_fraud_data(n_normal=500, n_fraud=50)

    results = run_fraud_pipeline(
        df,
        label_col=args.label_col,
        n_synthetic=args.n_synthetic,
        n_qubits=args.n_qubits,
        n_steps=args.n_steps,
    )
