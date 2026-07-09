"""
Sample Complexity Analyzer
===========================
Answers the key question: "What is the minimum number of training
samples needed for the IQP circuit to generate high-quality synthetic data?"

This is the core research contribution — finding the elbow point where
adding more training data stops improving synthetic data quality.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import warnings

from .evaluator import compute_mmd_squared


class SampleComplexityAnalyzer:
    """
    Analyzes how synthetic data quality varies with training dataset size.

    Trains IQP circuits at multiple dataset sizes and measures MMD²
    against the full class distribution. Finds the minimum dataset size
    where quality plateaus — the "elbow point".

    Parameters
    ----------
    dataset_sizes : list of int
        Training dataset sizes to evaluate.
        Recommended: [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
    n_seeds : int
        Number of random subsamples per (class, dataset_size).
        Higher = more reliable estimates but much slower.
        Recommended: 3–5 for exploratory, 10 for publication.
    ratio_threshold : float
        A dataset size is "sufficient" if MMD²(n) / MMD²(n_max) < ratio_threshold.
        Default 1.5 means within 50% of best possible quality.
    sigma : float
        MMD² bandwidth. Should match the generator's sigma.

    Examples
    --------
    >>> analyzer = SampleComplexityAnalyzer(dataset_sizes=[10, 50, 100, 250])
    >>> results = analyzer.run(X_binary, y, generator_config)
    >>> analyzer.plot(results)
    >>> min_n = analyzer.get_minimum_n(results)
    """

    def __init__(
        self,
        dataset_sizes: Optional[List[int]] = None,
        n_seeds: int = 3,
        ratio_threshold: float = 1.5,
        sigma: float = 3.0,
    ):
        if dataset_sizes is None:
            dataset_sizes = [10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500]
        self.dataset_sizes = sorted(dataset_sizes)
        self.n_seeds = n_seeds
        self.ratio_threshold = ratio_threshold
        self.sigma = sigma

    def run(
        self,
        X_binary: np.ndarray,
        y: np.ndarray,
        generator_config: Optional[Dict] = None,
        verbose: bool = True,
    ) -> Dict:
        """
        Run sample complexity analysis.

        Parameters
        ----------
        X_binary : ndarray (n_samples, n_qubits), binary
        y : ndarray (n_samples,)
        generator_config : dict, optional
            Config passed to IQPFinanceGenerator. Keys: n_qubits, local_gates,
            sigma, lr, n_steps, n_ops, n_samples, n_shots.
        verbose : bool

        Returns
        -------
        dict with keys: mmd_mean, mmd_std, mmd_all_runs, unique_counts,
                        minimum_n, ratios
        """
        try:
            from .generator import IQPFinanceGenerator
        except ImportError:
            raise ImportError("iqpopt required for sample complexity analysis.")

        if generator_config is None:
            generator_config = {}

        n_qubits = X_binary.shape[1]
        generator_config.setdefault("n_qubits", n_qubits)
        generator_config.setdefault("verbose", False)

        classes = np.unique(y)
        X_by_class = {c: X_binary[y == c] for c in classes}

        mmd_mean = {c: {} for c in classes}
        mmd_std  = {c: {} for c in classes}
        mmd_all  = {c: {} for c in classes}

        n_max = max(self.dataset_sizes)

        if verbose:
            print(f"Sample Complexity Analysis")
            print(f"{'='*50}")
            print(f"  dataset_sizes : {self.dataset_sizes}")
            print(f"  n_seeds       : {self.n_seeds}")
            print(f"  ratio_thresh  : {self.ratio_threshold}")
            print(f"  n_max         : {n_max}")
            print(f"{'='*50}\n")

        for dataset_size in self.dataset_sizes:
            if verbose:
                print(f"=== n = {dataset_size} ===")

            for c in classes:
                X_full = X_by_class[c]
                if len(X_full) < dataset_size:
                    if verbose:
                        print(f"  Class {c}: not enough samples, skipping.")
                    mmd_mean[c][dataset_size] = np.nan
                    mmd_std[c][dataset_size]  = np.nan
                    mmd_all[c][dataset_size]  = []
                    continue

                seed_mmds = []
                for seed in range(self.n_seeds):
                    rng = np.random.default_rng(seed * 1000 + int(c) * 100 + dataset_size)
                    idx = rng.choice(len(X_full), dataset_size, replace=False)
                    X_train = X_full[idx]

                    y_train = np.full(dataset_size, c)
                    cfg = {**generator_config, "random_seed": seed * 100 + int(c)}
                    gen = IQPFinanceGenerator(**cfg)

                    try:
                        gen.fit(X_train, y_train)
                        X_gen, _ = gen.sample(n_samples=500, class_label=c)
                        mmd_val = compute_mmd_squared(X_full, X_gen, self.sigma)
                        seed_mmds.append(mmd_val)
                        if verbose:
                            print(f"  Class {c} | seed {seed+1}/{self.n_seeds} | MMD²={mmd_val:.5f}")
                    except Exception as e:
                        warnings.warn(f"Class {c} seed {seed} failed: {e}")

                if seed_mmds:
                    mmd_mean[c][dataset_size] = float(np.mean(seed_mmds))
                    mmd_std[c][dataset_size]  = float(np.std(seed_mmds))
                    mmd_all[c][dataset_size]  = seed_mmds
                    if verbose:
                        print(f"  → Class {c} | mean={mmd_mean[c][dataset_size]:.5f} "
                              f"± {mmd_std[c][dataset_size]:.5f}\n")
                else:
                    mmd_mean[c][dataset_size] = np.nan
                    mmd_std[c][dataset_size]  = np.nan
                    mmd_all[c][dataset_size]  = []

        # Compute ratios and minimum n
        ratios = {c: {} for c in classes}
        minimum_n = {}

        for c in classes:
            val_max = mmd_mean[c].get(n_max, np.nan)
            found = False
            for ds in self.dataset_sizes:
                val = mmd_mean[c].get(ds, np.nan)
                if not np.isnan(val) and not np.isnan(val_max) and val_max > 0:
                    ratios[c][ds] = val / val_max
                    if not found and ratios[c][ds] < self.ratio_threshold:
                        minimum_n[c] = ds
                        found = True
                else:
                    ratios[c][ds] = np.nan
            if not found:
                minimum_n[c] = None

        if verbose:
            print("\n--- Minimum Sufficient Dataset Size ---")
            for c in classes:
                mn = minimum_n[c]
                if mn:
                    val = mmd_mean[c][mn]
                    val_max = mmd_mean[c].get(n_max, np.nan)
                    ratio = val / val_max if val_max > 0 else np.nan
                    print(f"  Class {c}: n = {mn}  "
                          f"(ratio={ratio:.2f}x  MMD²={val:.5f} vs {val_max:.5f} at n={n_max})")
                else:
                    print(f"  Class {c}: no sufficient n found in range")

        return {
            "mmd_mean": mmd_mean,
            "mmd_std":  mmd_std,
            "mmd_all":  mmd_all,
            "ratios":   ratios,
            "minimum_n": minimum_n,
            "dataset_sizes": self.dataset_sizes,
            "n_max": n_max,
            "ratio_threshold": self.ratio_threshold,
        }

    def plot(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        figsize: Tuple = (15, 6),
    ):
        """
        Plot sample complexity curves — MMD² vs dataset size with error bands
        and elbow markers.
        """
        colors  = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]
        markers = ["o", "s", "^", "D"]

        classes = list(results["mmd_mean"].keys())
        dataset_sizes = results["dataset_sizes"]
        n_max = results["n_max"]
        ratio_threshold = results["ratio_threshold"]

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(
            "Sample Complexity Analysis — IQP Finance Generator\n"
            "How much training data do you actually need?",
            fontsize=13, fontweight="bold",
        )

        # Left: MMD² ± std vs log(n)
        ax1 = axes[0]
        for i, c in enumerate(classes):
            y_mean = np.array([results["mmd_mean"][c].get(ds, np.nan) for ds in dataset_sizes])
            y_std  = np.array([results["mmd_std"][c].get(ds,  np.nan) for ds in dataset_sizes])
            ax1.plot(dataset_sizes, y_mean,
                     marker=markers[i % len(markers)],
                     color=colors[i % len(colors)],
                     linewidth=2, markersize=8, label=f"Class {c}")
            ax1.fill_between(dataset_sizes,
                             np.maximum(y_mean - y_std, 0), y_mean + y_std,
                             alpha=0.15, color=colors[i % len(colors)])
            # Elbow marker
            min_n = results["minimum_n"].get(c)
            if min_n is not None:
                val = results["mmd_mean"][c].get(min_n, np.nan)
                if not np.isnan(val):
                    ax1.scatter([min_n], [val],
                                color=colors[i % len(colors)],
                                s=150, zorder=6, marker="*",
                                edgecolors="black", linewidths=0.8)

        ax1.set_xscale("log")
        ax1.set_xlabel("Training Dataset Size (log scale)", fontsize=12)
        ax1.set_ylabel("MMD² (lower = better)", fontsize=12)
        ax1.set_title(f"MMD² ± std ({results.get('n_seeds', '?')} seeds)\n"
                      "★ = minimum sufficient n", fontsize=11)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.35, which="both")
        ax1.set_xticks(dataset_sizes)
        ax1.set_xticklabels([str(d) for d in dataset_sizes], rotation=30)
        ax1.set_ylim(bottom=0)

        # Right: Ratio plot
        ax2 = axes[1]
        ax2.axhline(ratio_threshold, color="red", linewidth=1.5, linestyle="--",
                    label=f"Redundancy threshold ({ratio_threshold}x)")
        ax2.axhline(1.0, color="green", linewidth=1.0, linestyle="--",
                    alpha=0.6, label=f"Same as n={n_max}")
        ax2.fill_between(dataset_sizes,
                         [1.0] * len(dataset_sizes),
                         [ratio_threshold] * len(dataset_sizes),
                         alpha=0.08, color="green", label="Acceptable zone")

        for i, c in enumerate(classes):
            ratios = [results["ratios"][c].get(ds, np.nan) for ds in dataset_sizes]
            ax2.plot(dataset_sizes, ratios,
                     marker=markers[i % len(markers)],
                     color=colors[i % len(colors)],
                     linewidth=2, markersize=8, label=f"Class {c}")

        ax2.set_xscale("log")
        ax2.set_xlabel("Training Dataset Size (log scale)", fontsize=12)
        ax2.set_ylabel(f"MMD²(n) / MMD²(n={n_max})", fontsize=12)
        ax2.set_title(f"Quality Ratio vs n={n_max}\n"
                      "Below red line = sufficient quality", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.35, which="both")
        ax2.set_xticks(dataset_sizes)
        ax2.set_xticklabels([str(d) for d in dataset_sizes], rotation=30)
        ax2.set_ylim(bottom=0)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        return fig

    def get_minimum_n(self, results: Dict) -> Dict:
        """Return the minimum sufficient dataset size per class."""
        return results["minimum_n"]
