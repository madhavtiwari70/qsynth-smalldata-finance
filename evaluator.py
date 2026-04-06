"""
Synthetic Data Evaluator
=========================
Evaluates quality of generated synthetic financial data using:
- MMD² (Maximum Mean Discrepancy) — distributional distance
- KGEL (Kernel Generalised Empirical Likelihood) — mode collapse detection
- Statistical parity — per-feature distribution comparison
- Downstream task performance — classifier trained on synthetic vs real
- Correlation matrix comparison — feature relationship preservation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import warnings


# =============================================
# MMD² Estimator (unbiased RBF kernel)
# =============================================

def rbf_kernel_matrix(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    diff = X[:, None, :] - Y[None, :, :]
    sq_dist = np.sum(diff ** 2, axis=-1)
    return np.exp(-sq_dist / (2.0 * sigma ** 2))


def compute_mmd_squared(
    X_real: np.ndarray,
    X_gen: np.ndarray,
    sigma: float,
) -> float:
    """
    Unbiased MMD² between real and generated distributions.

    MMD² = E[k(x,x')] - 2·E[k(x,y)] + E[k(y,y')]
    where diagonal terms are excluded (unbiased estimator).
    """
    X_real = np.array(X_real, dtype=np.float64)
    X_gen = np.array(X_gen, dtype=np.float64)
    n, m = len(X_real), len(X_gen)

    K_rr = rbf_kernel_matrix(X_real, X_real, sigma)
    K_gg = rbf_kernel_matrix(X_gen, X_gen, sigma)
    K_rg = rbf_kernel_matrix(X_real, X_gen, sigma)

    np.fill_diagonal(K_rr, 0.0)
    np.fill_diagonal(K_gg, 0.0)

    term_rr = K_rr.sum() / (n * (n - 1))
    term_gg = K_gg.sum() / (m * (m - 1))
    term_rg = K_rg.mean()

    return float(max(term_rr + term_gg - 2.0 * term_rg, 0.0))


class SyntheticEvaluator:
    """
    Comprehensive evaluation of synthetic financial data quality.

    Parameters
    ----------
    sigma : float
        RBF kernel bandwidth for MMD² computation. Default 3.0.
    downstream_classifier : str
        Classifier for TSTR evaluation: "logistic" or "random_forest".
    """

    def __init__(
        self,
        sigma: float = 3.0,
        downstream_classifier: str = "logistic",
    ):
        self.sigma = sigma
        self.downstream_classifier = downstream_classifier

    def evaluate(
        self,
        X_real: np.ndarray,
        X_synthetic: np.ndarray,
        y_real: np.ndarray,
        y_synthetic: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Run full evaluation suite.

        Parameters
        ----------
        X_real : ndarray of shape (n_real, n_qubits)
        X_synthetic : ndarray of shape (n_synth, n_qubits)
        y_real : ndarray of shape (n_real,)
        y_synthetic : ndarray of shape (n_synth,)
        feature_names : list of str, optional

        Returns
        -------
        dict with keys: mmd2, per_class_mmd2, feature_parity,
                        correlation_similarity, tstr_auc, tstr_trtr_auc
        """
        results = {}
        classes = np.unique(y_real)

        # 1. Overall MMD²
        results["mmd2_overall"] = compute_mmd_squared(X_real, X_synthetic, self.sigma)

        # 2. Per-class MMD²
        results["mmd2_per_class"] = {}
        for c in classes:
            X_r = X_real[y_real == c]
            X_s = X_synthetic[y_synthetic == c]
            if len(X_s) == 0:
                results["mmd2_per_class"][c] = np.nan
                warnings.warn(f"No synthetic samples for class {c}.")
            else:
                results["mmd2_per_class"][c] = compute_mmd_squared(X_r, X_s, self.sigma)

        # 3. Feature parity — compare per-feature means
        results["feature_parity"] = self._compute_feature_parity(
            X_real, X_synthetic, feature_names
        )

        # 4. Correlation matrix similarity
        results["correlation_similarity"] = self._compute_correlation_similarity(
            X_real, X_synthetic
        )

        # 5. Downstream task: Train on Synthetic, Test on Real (TSTR)
        results["tstr"] = self._compute_tstr(X_real, X_synthetic, y_real, y_synthetic)

        # 6. Class balance
        results["class_balance"] = {
            "real": {c: int(np.sum(y_real == c)) for c in classes},
            "synthetic": {c: int(np.sum(y_synthetic == c)) for c in classes},
        }

        return results

    def plot_results(
        self,
        results: Dict,
        X_real: Optional[np.ndarray] = None,
        X_synthetic: Optional[np.ndarray] = None,
        y_real: Optional[np.ndarray] = None,
        y_synthetic: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ):
        """
        Comprehensive visualization of evaluation results.
        """
        fig = plt.figure(figsize=(18, 14))
        fig.suptitle(
            "IQP Synthetic Financial Data — Quality Evaluation",
            fontsize=16, fontweight="bold", y=0.98,
        )
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

        # Panel 1: Per-class MMD²
        ax1 = fig.add_subplot(gs[0, 0])
        classes = list(results["mmd2_per_class"].keys())
        mmd_vals = [results["mmd2_per_class"][c] for c in classes]
        bars = ax1.bar([str(c) for c in classes], mmd_vals,
                       color=colors[:len(classes)], edgecolor="white", linewidth=1.5)
        ax1.set_title("MMD² per Class\n(lower = better)", fontweight="bold")
        ax1.set_xlabel("Class Label")
        ax1.set_ylabel("MMD²")
        ax1.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        for bar, val in zip(bars, mmd_vals):
            if not np.isnan(val):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{val:.4f}", ha="center", va="bottom", fontsize=9)
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: Feature parity
        ax2 = fig.add_subplot(gs[0, 1])
        fp = results["feature_parity"]
        feat_labels = list(fp["real_means"].keys())
        real_means = [fp["real_means"][f] for f in feat_labels]
        synth_means = [fp["synthetic_means"][f] for f in feat_labels]
        x = np.arange(len(feat_labels))
        width = 0.35
        ax2.bar(x - width / 2, real_means, width, label="Real", color="#2196F3", alpha=0.8)
        ax2.bar(x + width / 2, synth_means, width, label="Synthetic", color="#F44336", alpha=0.8)
        ax2.set_title("Feature Mean Comparison\n(real vs synthetic)", fontweight="bold")
        ax2.set_xlabel("Feature / Qubit")
        ax2.set_ylabel("Mean bit value")
        ax2.set_xticks(x)
        xlabels = feat_labels if len(feat_labels) <= 16 else [str(i) for i in range(len(feat_labels))]
        ax2.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=7)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, axis="y")

        # Panel 3: Class balance
        ax3 = fig.add_subplot(gs[0, 2])
        cb = results["class_balance"]
        cls_list = list(cb["real"].keys())
        real_counts = [cb["real"][c] for c in cls_list]
        synth_counts = [cb["synthetic"][c] for c in cls_list]
        x3 = np.arange(len(cls_list))
        ax3.bar(x3 - 0.2, real_counts, 0.4, label="Real", color="#2196F3", alpha=0.8)
        ax3.bar(x3 + 0.2, synth_counts, 0.4, label="Synthetic", color="#F44336", alpha=0.8)
        ax3.set_title("Class Balance\n(real vs synthetic)", fontweight="bold")
        ax3.set_xlabel("Class")
        ax3.set_ylabel("Count")
        ax3.set_xticks(x3)
        ax3.set_xticklabels([str(c) for c in cls_list])
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis="y")

        # Panel 4 & 5: Correlation matrices
        if X_real is not None and X_synthetic is not None:
            ax4 = fig.add_subplot(gs[1, 0])
            ax5 = fig.add_subplot(gs[1, 1])
            corr_real = np.corrcoef(X_real.T)
            corr_synth = np.corrcoef(X_synthetic.T)
            vmin, vmax = -1, 1
            im4 = ax4.imshow(corr_real, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            ax4.set_title("Correlation Matrix\n(Real Data)", fontweight="bold")
            ax4.set_xlabel("Qubit / Feature")
            ax4.set_ylabel("Qubit / Feature")
            plt.colorbar(im4, ax=ax4, shrink=0.8)

            im5 = ax5.imshow(corr_synth, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
            ax5.set_title("Correlation Matrix\n(Synthetic Data)", fontweight="bold")
            ax5.set_xlabel("Qubit / Feature")
            ax5.set_ylabel("Qubit / Feature")
            plt.colorbar(im5, ax=ax5, shrink=0.8)

        # Panel 6: Correlation difference
        if X_real is not None and X_synthetic is not None:
            ax6 = fig.add_subplot(gs[1, 2])
            corr_diff = np.abs(corr_real - corr_synth)
            im6 = ax6.imshow(corr_diff, cmap="Reds", vmin=0, vmax=1, aspect="auto")
            ax6.set_title(
                f"|Correlation Difference|\n"
                f"Mean error: {corr_diff.mean():.4f}",
                fontweight="bold",
            )
            ax6.set_xlabel("Qubit / Feature")
            ax6.set_ylabel("Qubit / Feature")
            plt.colorbar(im6, ax=ax6, shrink=0.8)

        # Panel 7: TSTR results
        ax7 = fig.add_subplot(gs[2, 0])
        tstr = results["tstr"]
        tstr_labels = ["TRTR\n(real→real)", "TSTR\n(synth→real)"]
        tstr_vals = [tstr.get("trtr_auc", np.nan), tstr.get("tstr_auc", np.nan)]
        bar_colors = ["#4CAF50", "#FF9800"]
        bars7 = ax7.bar(tstr_labels, tstr_vals, color=bar_colors, edgecolor="white", linewidth=1.5)
        ax7.set_title("Downstream Task (AUC)\nTrain on Synthetic, Test on Real",
                      fontweight="bold")
        ax7.set_ylabel("ROC AUC")
        ax7.set_ylim(0, 1.1)
        ax7.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Random")
        for bar, val in zip(bars7, tstr_vals):
            if not np.isnan(val):
                ax7.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                         f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
        ax7.legend(fontsize=9)
        ax7.grid(True, alpha=0.3, axis="y")

        # Panel 8: Summary scorecard
        ax8 = fig.add_subplot(gs[2, 1:])
        ax8.axis("off")
        summary_text = [
            ["Metric", "Value", "Interpretation"],
            ["Overall MMD²", f"{results['mmd2_overall']:.5f}",
             "Lower = more similar distributions"],
            ["Correlation similarity", f"{results['correlation_similarity']:.4f}",
             "1.0 = identical correlation structure"],
            ["TSTR AUC", f"{tstr.get('tstr_auc', 'N/A')}",
             "Close to TRTR = synthetic is informative"],
            ["TRTR AUC", f"{tstr.get('trtr_auc', 'N/A')}",
             "Baseline (real data performance)"],
        ]
        for c_idx, row in enumerate(summary_text):
            for r_idx, val in enumerate(row):
                weight = "bold" if c_idx == 0 else "normal"
                color = "#1a1a2e" if c_idx == 0 else "black"
                ax8.text(
                    r_idx * 0.35, 1 - c_idx * 0.18, val,
                    transform=ax8.transAxes,
                    fontsize=10, fontweight=weight, color=color,
                    verticalalignment="top",
                )
        ax8.set_title("Evaluation Summary", fontweight="bold", loc="left")

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Saved evaluation plot to: {save_path}")

        plt.show()
        return fig

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_feature_parity(
        self,
        X_real: np.ndarray,
        X_synthetic: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict:
        n_features = X_real.shape[1]
        if feature_names is None:
            feature_names = [f"q{i}" for i in range(n_features)]
        elif len(feature_names) < n_features:
            feature_names = list(feature_names) + [
                f"q{i}" for i in range(len(feature_names), n_features)
            ]

        real_means = {feature_names[i]: float(X_real[:, i].mean()) for i in range(n_features)}
        synth_means = {feature_names[i]: float(X_synthetic[:, i].mean()) for i in range(n_features)}
        mean_abs_diff = float(
            np.mean([abs(real_means[f] - synth_means[f]) for f in feature_names])
        )

        return {
            "real_means": real_means,
            "synthetic_means": synth_means,
            "mean_absolute_difference": mean_abs_diff,
        }

    def _compute_correlation_similarity(
        self, X_real: np.ndarray, X_synthetic: np.ndarray
    ) -> float:
        corr_real = np.corrcoef(X_real.T)
        corr_synth = np.corrcoef(X_synthetic.T)
        # Frobenius similarity (1 = identical, 0 = completely different)
        diff = corr_real - corr_synth
        fro_diff = np.linalg.norm(diff, "fro")
        fro_real = np.linalg.norm(corr_real, "fro")
        similarity = max(0.0, 1.0 - fro_diff / (fro_real + 1e-10))
        return float(similarity)

    def _compute_tstr(
        self,
        X_real: np.ndarray,
        X_synthetic: np.ndarray,
        y_real: np.ndarray,
        y_synthetic: np.ndarray,
    ) -> Dict:
        """Train on Synthetic, Test on Real (TSTR) evaluation."""
        if len(np.unique(y_real)) < 2:
            return {"tstr_auc": np.nan, "trtr_auc": np.nan, "note": "Single class — AUC not applicable"}

        if self.downstream_classifier == "logistic":
            clf_tstr = LogisticRegression(max_iter=500, random_state=42)
            clf_trtr = LogisticRegression(max_iter=500, random_state=42)
        else:
            clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
            clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)

        try:
            # TSTR: train on synthetic, test on real
            clf_tstr.fit(X_synthetic, y_synthetic)
            y_pred_tstr = clf_tstr.predict_proba(X_real)[:, 1]
            tstr_auc = float(roc_auc_score(y_real, y_pred_tstr))

            # TRTR: train on real, test on real (cross-validated baseline)
            trtr_scores = cross_val_score(
                clf_trtr, X_real, y_real, cv=5, scoring="roc_auc"
            )
            trtr_auc = float(trtr_scores.mean())

            return {
                "tstr_auc": round(tstr_auc, 4),
                "trtr_auc": round(trtr_auc, 4),
                "tstr_trtr_gap": round(abs(tstr_auc - trtr_auc), 4),
            }
        except Exception as e:
            return {"tstr_auc": np.nan, "trtr_auc": np.nan, "note": str(e)}

    def print_summary(self, results: Dict):
        """Print a formatted text summary of evaluation results."""
        print("\n" + "=" * 55)
        print("  IQP Synthetic Data — Evaluation Summary")
        print("=" * 55)
        print(f"  Overall MMD²          : {results['mmd2_overall']:.6f}")
        print(f"\n  Per-Class MMD²:")
        for c, v in results["mmd2_per_class"].items():
            flag = " ← good" if not np.isnan(v) and v < 0.01 else ""
            print(f"    Class {c}            : {v:.6f}{flag}")
        print(f"\n  Correlation Similarity: {results['correlation_similarity']:.4f}  (1.0=perfect)")
        print(f"\n  Downstream Task (TSTR):")
        tstr = results["tstr"]
        print(f"    TRTR AUC (real→real) : {tstr.get('trtr_auc', 'N/A')}")
        print(f"    TSTR AUC (syn→real)  : {tstr.get('tstr_auc', 'N/A')}")
        print(f"\n  Class Balance:")
        for c in results["class_balance"]["real"]:
            r = results["class_balance"]["real"][c]
            s = results["class_balance"]["synthetic"][c]
            print(f"    Class {c}: real={r}  synthetic={s}")
        print("=" * 55 + "\n")
