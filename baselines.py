"""
Classical Baselines
====================
Bernoulli and RBM baselines for comparing against IQP circuit performance.
These are used to demonstrate whether the quantum model adds value over
the simplest classical alternatives.
"""

import numpy as np
from typing import Dict, Optional, Tuple
import warnings

try:
    from sklearn.neural_network import BernoulliRBM
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class BernoulliBaseline:
    """
    Independent Bernoulli baseline: samples each bit independently
    from its empirical marginal probability.

    This is the simplest possible synthetic data generator — it captures
    per-feature marginals but no correlations between features.
    Comparing IQP results against this baseline shows whether the circuit
    is learning feature correlations or just marginal statistics.

    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility.
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.classes_: Optional[np.ndarray] = None
        self.class_probs_: Optional[Dict] = None
        self.class_weights_: Optional[Dict] = None
        self.is_fitted_ = False
        # A single RNG instance advances across calls, so repeated calls to
        # sample() draw new data instead of replaying the same output.
        self._rng = np.random.default_rng(random_seed)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BernoulliBaseline":
        """
        Fit per-class marginal probabilities.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features), binary
        y : ndarray of shape (n_samples,)
        """
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.class_probs_ = {}
        self.class_weights_ = {}
        total = len(y)

        for c in self.classes_:
            X_c = X[y == c]
            # Per-feature marginal probability of being 1
            self.class_probs_[c] = X_c.mean(axis=0)
            self.class_weights_[c] = len(X_c) / total

        self.is_fitted_ = True
        return self

    def sample(
        self,
        n_samples: int = 100,
        class_label=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples by independent Bernoulli draws.

        Parameters
        ----------
        n_samples : int
        class_label : optional, generate from single class only

        Returns
        -------
        X_synthetic : ndarray (n_samples, n_features)
        y_synthetic : ndarray (n_samples,)
        """
        self._check_fitted()

        if class_label is not None:
            labels_to_sample = {class_label: n_samples}
        else:
            labels_to_sample = {}
            remaining = n_samples
            for i, c in enumerate(sorted(self.classes_)):
                if i < len(self.classes_) - 1:
                    n_c = int(n_samples * self.class_weights_[c])
                    labels_to_sample[c] = n_c
                    remaining -= n_c
                else:
                    labels_to_sample[c] = remaining

        all_X, all_y = [], []
        for c, n_c in labels_to_sample.items():
            if n_c == 0:
                continue
            probs = self.class_probs_[c]
            X_c = (self._rng.random((n_c, len(probs))) < probs).astype(float)
            all_X.append(X_c)
            all_y.append(np.full(n_c, c))

        X_synthetic = np.vstack(all_X)
        y_synthetic = np.concatenate(all_y)
        perm = self._rng.permutation(len(X_synthetic))
        return X_synthetic[perm], y_synthetic[perm]

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("BernoulliBaseline is not fitted. Call fit() first.")


class RBMBaseline:
    """
    Restricted Boltzmann Machine baseline using scikit-learn's BernoulliRBM.

    Captures pairwise correlations via hidden units. More expressive than
    Bernoulli baseline but requires more data to train reliably.

    Parameters
    ----------
    n_components : int
        Number of hidden units. More = more expressive but needs more data.
    n_iter : int
        Training epochs.
    learning_rate : float
        RBM learning rate.
    random_seed : int
    """

    def __init__(
        self,
        n_components: int = 64,
        n_iter: int = 100,
        learning_rate: float = 0.01,
        random_seed: int = 42,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for RBMBaseline.")

        self.n_components = n_components
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.classes_: Optional[np.ndarray] = None
        self.rbms_: Optional[Dict] = None
        self.class_weights_: Optional[Dict] = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RBMBaseline":
        """Fit one RBM per class."""
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.rbms_ = {}
        self.class_weights_ = {}
        total = len(y)

        for c in self.classes_:
            X_c = X[y == c]
            self.class_weights_[c] = len(X_c) / total

            rbm = BernoulliRBM(
                n_components=self.n_components,
                n_iter=self.n_iter,
                learning_rate=self.learning_rate,
                random_state=self.random_seed,
                verbose=False,
            )
            rbm.fit(X_c)
            self.rbms_[c] = rbm

        self.is_fitted_ = True
        return self

    def sample(
        self,
        n_samples: int = 100,
        n_gibbs_steps: int = 1000,
        class_label=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate samples via Gibbs sampling from each RBM.

        Parameters
        ----------
        n_samples : int
        n_gibbs_steps : int
            Number of Gibbs steps per sample (more = better quality but slower).
        class_label : optional
        """
        self._check_fitted()

        if class_label is not None:
            labels_to_sample = {class_label: n_samples}
        else:
            labels_to_sample = {}
            remaining = n_samples
            for i, c in enumerate(sorted(self.classes_)):
                if i < len(self.classes_) - 1:
                    n_c = int(n_samples * self.class_weights_[c])
                    labels_to_sample[c] = n_c
                    remaining -= n_c
                else:
                    labels_to_sample[c] = remaining

        all_X, all_y = [], []
        rng = np.random.default_rng(self.random_seed)

        for c, n_c in labels_to_sample.items():
            if n_c == 0:
                continue
            rbm = self.rbms_[c]
            n_visible = rbm.components_.shape[1]

            # Initialize from random binary state
            v = (rng.random((n_c, n_visible)) > 0.5).astype(float)

            # Gibbs sampling
            for _ in range(n_gibbs_steps):
                # Sample hidden given visible
                p_h = self._sigmoid(v @ rbm.components_.T + rbm.intercept_hidden_)
                h = (rng.random(p_h.shape) < p_h).astype(float)
                # Sample visible given hidden
                p_v = self._sigmoid(h @ rbm.components_ + rbm.intercept_visible_)
                v = (rng.random(p_v.shape) < p_v).astype(float)

            all_X.append(v)
            all_y.append(np.full(n_c, c))

        X_synthetic = np.vstack(all_X)
        y_synthetic = np.concatenate(all_y)
        perm = rng.permutation(len(X_synthetic))
        return X_synthetic[perm], y_synthetic[perm]

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("RBMBaseline is not fitted. Call fit() first.")
