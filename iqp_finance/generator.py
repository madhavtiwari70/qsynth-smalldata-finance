"""
IQP Finance Generator
======================
Trains per-class IQP circuits on binary financial data using MMD loss,
then generates synthetic binary samples that can be decoded back to
financial feature values.

Based on: Recio-Armengol et al. (2025) arXiv:2503.02934
Implementation uses IQPopt package for efficient classical training.
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Union
import warnings

try:
    import iqpopt as iqp
    import iqpopt.gen_qml as gen
    from iqpopt.utils import local_gates
    IQP_AVAILABLE = True
except ImportError:
    IQP_AVAILABLE = False
    warnings.warn(
        "iqpopt not found. Install with: pip install iqpopt\n"
        "Generator will raise an error when fit() is called.",
        ImportWarning,
    )


class IQPFinanceGenerator:
    """
    Quantum generative model for financial synthetic data using IQP circuits.

    Trains one IQP circuit per class label using the MMD² loss function.
    Training is done entirely on classical hardware via efficient simulation.
    Sampling can be done classically (for small circuits) or on quantum hardware.

    Parameters
    ----------
    n_qubits : int
        Number of qubits = dimensionality of binary input vectors.
    local_gates : int
        Pauli weight of gate generators. Controls expressibility vs trainability:
        - 2: pairwise correlations only, fast training, limited expressibility
        - 3: 3-way correlations, recommended for n_qubits=16 (sweet spot)
        - 4: 4-way correlations, harder to train, needs more steps
    sigma : float
        MMD bandwidth. Controls which order of correlations the loss probes.
        Recommended: 3.0 for binary 16-dim vectors.
        Lower sigma probes higher-order correlations.
    lr : float
        Adam learning rate. Default 0.001 (prevents mode collapse).
    n_steps : int
        Number of training steps per class. Default 1000.
    n_ops : int
        Number of operator samples for MMD gradient estimation.
        Higher = lower variance gradients but slower. Default 200.
    n_samples : int
        Number of bitstring samples for expectation value estimation. Default 500.
    n_shots : int
        Number of shots when generating synthetic samples. Default 2000.
    collapse_threshold : int
        Minimum unique samples in n_shots to not flag as mode collapse.
        Default 500 (25% of n_shots=2000).
    random_seed : int
        Base random seed for reproducibility.
    verbose : bool
        Print training progress per class.

    Examples
    --------
    >>> gen = IQPFinanceGenerator(n_qubits=16, local_gates=3)
    >>> gen.fit(X_binary, y_labels)
    >>> X_synth, y_synth = gen.sample(n_samples=200)
    """

    def __init__(
        self,
        n_qubits: int = 16,
        local_gates: int = 3,
        sigma: float = 3.0,
        lr: float = 0.001,
        n_steps: int = 1000,
        n_ops: int = 200,
        n_samples: int = 500,
        n_shots: int = 2000,
        collapse_threshold: int = 500,
        random_seed: int = 42,
        verbose: bool = True,
    ):
        if not IQP_AVAILABLE:
            raise ImportError("iqpopt is required. Install with: pip install iqpopt")

        self.n_qubits = n_qubits
        self.local_gates_k = local_gates
        self.sigma = sigma
        self.lr = lr
        self.n_steps = n_steps
        self.n_ops = n_ops
        self.n_samples = n_samples
        self.n_shots = n_shots
        self.collapse_threshold = collapse_threshold
        self.random_seed = random_seed
        self.verbose = verbose
        # Persistent RNG for sampling, so runs are reproducible from
        # random_seed but successive sample() calls still draw new data.
        self._sample_rng = np.random.default_rng(random_seed)

        # Fitted attributes
        self.classes_: Optional[np.ndarray] = None
        self.trained_params_: Optional[Dict] = None
        self.training_history_: Optional[Dict] = None
        self.collapse_flags_: Optional[Dict] = None
        self.gates_ = None
        self.is_fitted_ = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        class_weights: Optional[Dict] = None,
    ) -> "IQPFinanceGenerator":
        """
        Train one IQP circuit per class label.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_qubits)
            Binary feature matrix (values in {0, 1}).
        y : ndarray of shape (n_samples,)
            Class labels (e.g. 0=normal, 1=fraud).
        class_weights : dict, optional
            Mapping from class label to sampling weight when generating.
            If None, uses empirical class frequencies.

        Returns
        -------
        self
        """
        X = np.array(X, dtype=float)
        y = np.array(y)

        if X.shape[1] != self.n_qubits:
            raise ValueError(
                f"X has {X.shape[1]} features but n_qubits={self.n_qubits}. "
                "Run FinancialDataPreprocessor first."
            )
        if not np.all(np.isin(X, [0, 1])):
            raise ValueError("X must be binary (values in {0, 1}).")

        self.classes_ = np.unique(y)
        self.gates_ = local_gates(self.n_qubits, self.local_gates_k)
        self.trained_params_ = {}
        self.training_history_ = {}
        self.collapse_flags_ = {}

        # Compute class weights
        if class_weights is None:
            class_counts = {c: np.sum(y == c) for c in self.classes_}
            total = len(y)
            self.class_weights_ = {c: class_counts[c] / total for c in self.classes_}
        else:
            self.class_weights_ = class_weights

        n_params = len(self.gates_)
        if self.verbose:
            print(f"IQP Finance Generator — Training Configuration")
            print(f"{'=' * 50}")
            print(f"  n_qubits        : {self.n_qubits}")
            print(f"  local_gates k   : {self.local_gates_k}")
            print(f"  n_gates (params): {n_params}")
            print(f"  sigma           : {self.sigma}")
            print(f"  lr              : {self.lr}")
            print(f"  n_steps         : {self.n_steps}")
            print(f"  classes         : {self.classes_}")
            print(f"{'=' * 50}\n")

        for class_label in self.classes_:
            X_class = X[y == class_label]
            n_class = len(X_class)

            if self.verbose:
                print(f"Training Class {class_label} | n={n_class} samples")

            # JAX key splitting — prevents biased gradients
            master_key = jax.random.PRNGKey(
                self.random_seed + int(class_label) * 100
            )
            init_key, train_key = jax.random.split(master_key)

            circuit = iqp.IqpSimulator(self.n_qubits, self.gates_)
            params_init = jax.random.normal(init_key, shape=(n_params,)) * 0.1

            loss_kwargs = {
                "params": params_init,
                "iqp_circuit": circuit,
                "ground_truth": jnp.array(X_class),
                "sigma": [self.sigma],
                "n_ops": self.n_ops,
                "n_samples": self.n_samples,
                "key": train_key,
            }

            trainer = iqp.Trainer("Adam", gen.mmd_loss_iqp, self.lr)
            trainer.train(self.n_steps, loss_kwargs)

            self.trained_params_[class_label] = trainer.final_params
            self.training_history_[class_label] = {
                "final_loss": float(trainer.loss_history[-1]) if hasattr(trainer, "loss_history") else None,
                "n_train_samples": n_class,
            }

            if self.verbose:
                print(f"  ✓ Class {class_label} trained | "
                      f"final_loss={self.training_history_[class_label]['final_loss']}\n")

        self.is_fitted_ = True
        return self

    def sample(
        self,
        n_samples: int = 100,
        class_label: Optional[Union[int, str]] = None,
        return_decoded: bool = False,
        preprocessor=None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic binary samples from trained circuits.

        Parameters
        ----------
        n_samples : int
            Total number of synthetic samples to generate.
        class_label : int or str, optional
            If provided, generate samples only from this class circuit.
            If None, generates proportionally across all classes.
        return_decoded : bool
            If True and preprocessor is provided, returns decoded financial values.
        preprocessor : FinancialDataPreprocessor, optional
            Required if return_decoded=True.

        Returns
        -------
        X_synthetic : ndarray of shape (n_samples, n_qubits)
            Binary synthetic samples (or decoded DataFrame if return_decoded=True).
        y_synthetic : ndarray of shape (n_samples,)
            Class labels for each synthetic sample.
        """
        self._check_fitted()

        if class_label is not None:
            if class_label not in self.classes_:
                raise ValueError(f"class_label {class_label} not seen during training.")
            labels_to_sample = {class_label: n_samples}
        else:
            # Sample proportionally to class weights
            labels_to_sample = {}
            remaining = n_samples
            sorted_classes = sorted(self.classes_, key=lambda c: self.class_weights_[c])
            for i, c in enumerate(sorted_classes[:-1]):
                n_c = int(n_samples * self.class_weights_[c])
                labels_to_sample[c] = n_c
                remaining -= n_c
            labels_to_sample[sorted_classes[-1]] = remaining

        all_X = []
        all_y = []

        for c, n_c in labels_to_sample.items():
            if n_c == 0:
                continue

            circuit = iqp.IqpSimulator(self.n_qubits, self.gates_)
            shots_output = circuit.sample(self.trained_params_[c], shots=self.n_shots)
            X_pool = np.array(shots_output).reshape(-1, self.n_qubits)

            # Check for mode collapse
            n_unique = len(np.unique(X_pool, axis=0))
            self.collapse_flags_[c] = n_unique < self.collapse_threshold
            if self.collapse_flags_[c]:
                warnings.warn(
                    f"Class {c}: possible mode collapse detected "
                    f"({n_unique}/{self.n_shots} unique samples). "
                    "Consider reducing lr or increasing n_steps.",
                    UserWarning,
                )

            # Subsample to requested count
            if n_c <= len(X_pool):
                idx = self._sample_rng.choice(len(X_pool), n_c, replace=False)
            else:
                idx = self._sample_rng.choice(len(X_pool), n_c, replace=True)
                warnings.warn(
                    f"Requested {n_c} samples but only {len(X_pool)} unique circuit "
                    "outputs — sampling with replacement.",
                    UserWarning,
                )
            all_X.append(X_pool[idx])
            all_y.append(np.full(n_c, c))

        X_synthetic = np.vstack(all_X)
        y_synthetic = np.concatenate(all_y)

        # Shuffle
        perm = self._sample_rng.permutation(len(X_synthetic))
        X_synthetic = X_synthetic[perm]
        y_synthetic = y_synthetic[perm]

        if return_decoded:
            if preprocessor is None:
                raise ValueError("preprocessor must be provided when return_decoded=True")
            X_synthetic = preprocessor.inverse_transform(X_synthetic)

        return X_synthetic, y_synthetic

    def get_training_summary(self) -> Dict:
        """Return a summary of training results per class."""
        self._check_fitted()
        return {
            "classes": list(self.classes_),
            "n_params": len(self.gates_),
            "training_history": self.training_history_,
            "class_weights": self.class_weights_,
        }

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Generator is not fitted. Call fit() first.")
