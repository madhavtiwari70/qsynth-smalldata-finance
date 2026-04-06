"""
Financial Data Preprocessor
============================
Converts continuous financial tabular data into binary vectors
suitable for IQP circuit input.

Supported binarization methods:
- quantile: discretize each feature into n_quantile bins
- threshold: binary threshold at median or custom value
- zscore: threshold at z=0 (above/below mean)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from typing import Optional, Union, Dict, List, Tuple
import warnings


class FinancialDataPreprocessor:
    """
    Preprocesses financial tabular data into binary vectors for IQP circuits.

    Parameters
    ----------
    n_qubits : int
        Total number of qubits in the IQP circuit. The binary vectors
        produced will have length n_qubits.
    binarize_method : str
        Method to binarize continuous features:
        - "quantile": discretize into n_quantile equal-frequency bins
        - "threshold": binary threshold at per-feature median
        - "zscore": binary threshold at feature mean (above=1, below=0)
    n_quantiles : int
        Number of quantile bins per feature (only used when
        binarize_method="quantile"). bits_per_feature = log2(n_quantiles).
        Must be a power of 2. Default 4 → 2 bits per feature.
    features : list of str, optional
        List of feature column names to use. If None, uses all numeric columns.

    Examples
    --------
    >>> preprocessor = FinancialDataPreprocessor(n_qubits=16, binarize_method="quantile")
    >>> X_binary, feature_map = preprocessor.fit_transform(df)
    >>> X_reconstructed = preprocessor.inverse_transform(X_binary)
    """

    SUPPORTED_METHODS = ["quantile", "threshold", "zscore"]

    def __init__(
        self,
        n_qubits: int = 16,
        binarize_method: str = "quantile",
        n_quantiles: int = 4,
        features: Optional[List[str]] = None,
    ):
        if binarize_method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"binarize_method must be one of {self.SUPPORTED_METHODS}, "
                f"got '{binarize_method}'"
            )
        if binarize_method == "quantile" and (n_quantiles & (n_quantiles - 1)) != 0:
            raise ValueError("n_quantiles must be a power of 2 (2, 4, 8, 16, ...)")

        self.n_qubits = n_qubits
        self.binarize_method = binarize_method
        self.n_quantiles = n_quantiles
        self.features = features

        # Computed during fit
        self.bits_per_feature_: Optional[int] = None
        self.n_features_used_: Optional[int] = None
        self.feature_names_: Optional[List[str]] = None
        self.feature_map_: Optional[Dict] = None
        self._discretizer: Optional[KBinsDiscretizer] = None
        self._medians: Optional[np.ndarray] = None
        self._means: Optional[np.ndarray] = None
        self._stds: Optional[np.ndarray] = None
        self._bin_edges_: Optional[List] = None
        self.is_fitted_ = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FinancialDataPreprocessor":
        """
        Fit the preprocessor to financial data.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)
            Financial feature matrix (numeric columns only).

        Returns
        -------
        self
        """
        X_df = self._validate_input(X)

        # Select features
        if self.features is not None:
            missing = set(self.features) - set(X_df.columns)
            if missing:
                raise ValueError(f"Features not found in data: {missing}")
            X_df = X_df[self.features]

        self.feature_names_ = list(X_df.columns)
        X_arr = X_df.values.astype(np.float64)

        # Determine bits per feature based on method
        if self.binarize_method == "quantile":
            self.bits_per_feature_ = int(np.log2(self.n_quantiles))
        else:
            self.bits_per_feature_ = 1

        # Compute how many features we can fit
        max_features = self.n_qubits // self.bits_per_feature_
        if X_arr.shape[1] > max_features:
            warnings.warn(
                f"Data has {X_arr.shape[1]} features but n_qubits={self.n_qubits} "
                f"only fits {max_features} features with {self.bits_per_feature_} "
                f"bits each. Using first {max_features} features.",
                UserWarning,
            )
            X_arr = X_arr[:, :max_features]
            self.feature_names_ = self.feature_names_[:max_features]

        self.n_features_used_ = X_arr.shape[1]
        total_bits = self.n_features_used_ * self.bits_per_feature_

        if total_bits < self.n_qubits:
            warnings.warn(
                f"Features produce {total_bits} bits but n_qubits={self.n_qubits}. "
                f"Remaining {self.n_qubits - total_bits} qubits will be padded with 0.",
                UserWarning,
            )

        # Fit binarizer
        if self.binarize_method == "quantile":
            self._discretizer = KBinsDiscretizer(
                n_bins=self.n_quantiles,
                encode="ordinal",
                strategy="quantile",
                subsample=None,
            )
            self._discretizer.fit(X_arr)
            self._bin_edges_ = self._discretizer.bin_edges_

        elif self.binarize_method == "threshold":
            self._medians = np.median(X_arr, axis=0)

        elif self.binarize_method == "zscore":
            self._means = np.mean(X_arr, axis=0)
            self._stds = np.std(X_arr, axis=0) + 1e-10

        # Build feature map for interpretability
        self.feature_map_ = self._build_feature_map()
        self.is_fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform financial data to binary vectors.

        Parameters
        ----------
        X : DataFrame or ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_binary : ndarray of shape (n_samples, n_qubits)
            Binary matrix with values in {0, 1}.
        """
        self._check_fitted()
        X_df = self._validate_input(X)

        if self.features is not None:
            X_df = X_df[self.features]

        X_arr = X_df[self.feature_names_].values.astype(np.float64)
        n_samples = X_arr.shape[0]

        if self.binarize_method == "quantile":
            # Discretize to bin indices then convert to binary
            bins = self._discretizer.transform(X_arr).astype(int)
            # Clip to valid range
            bins = np.clip(bins, 0, self.n_quantiles - 1)
            # Convert each bin index to binary representation
            X_binary_features = self._bins_to_binary(bins)

        elif self.binarize_method == "threshold":
            X_binary_features = (X_arr > self._medians).astype(float)

        elif self.binarize_method == "zscore":
            X_zscore = (X_arr - self._means) / self._stds
            X_binary_features = (X_zscore > 0).astype(float)

        # Pad to n_qubits if needed
        if X_binary_features.shape[1] < self.n_qubits:
            padding = np.zeros((n_samples, self.n_qubits - X_binary_features.shape[1]))
            X_binary_features = np.hstack([X_binary_features, padding])

        return X_binary_features[:, :self.n_qubits]

    def fit_transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Fit and transform in one step.

        Returns
        -------
        X_binary : ndarray of shape (n_samples, n_qubits)
        feature_map : dict describing which qubits correspond to which features
        """
        self.fit(X)
        return self.transform(X), self.feature_map_

    def inverse_transform(self, X_binary: np.ndarray) -> pd.DataFrame:
        """
        Convert binary vectors back to approximate financial values.

        Parameters
        ----------
        X_binary : ndarray of shape (n_samples, n_qubits)

        Returns
        -------
        DataFrame with reconstructed financial feature values.
        """
        self._check_fitted()
        n_samples = X_binary.shape[0]
        X_binary = X_binary[:, :self.n_features_used_ * self.bits_per_feature_]

        if self.binarize_method == "quantile":
            # Reconstruct bin indices from binary
            bins = self._binary_to_bins(X_binary)
            # Map bins to bin midpoints
            X_reconstructed = np.zeros((n_samples, self.n_features_used_))
            for feat_idx in range(self.n_features_used_):
                edges = self._bin_edges_[feat_idx]
                for sample_idx in range(n_samples):
                    bin_idx = int(bins[sample_idx, feat_idx])
                    bin_idx = min(bin_idx, len(edges) - 2)
                    midpoint = (edges[bin_idx] + edges[bin_idx + 1]) / 2.0
                    X_reconstructed[sample_idx, feat_idx] = midpoint

        elif self.binarize_method == "threshold":
            # Map 0/1 back to median ± small noise
            X_reconstructed = np.where(
                X_binary[:, :self.n_features_used_] == 1,
                self._medians + np.abs(self._medians) * 0.1,
                self._medians - np.abs(self._medians) * 0.1,
            )

        elif self.binarize_method == "zscore":
            # Map 0/1 back to mean ± 1 std
            X_reconstructed = np.where(
                X_binary[:, :self.n_features_used_] == 1,
                self._means + self._stds,
                self._means - self._stds,
            )

        return pd.DataFrame(X_reconstructed, columns=self.feature_names_)

    def get_density_per_feature(self, X_binary: np.ndarray) -> Dict[str, float]:
        """
        Compute the fraction of 1s per feature in the binary representation.
        Useful for diagnosing binarization quality.

        Parameters
        ----------
        X_binary : ndarray of shape (n_samples, n_qubits)

        Returns
        -------
        dict mapping feature name to average bit density
        """
        self._check_fitted()
        densities = {}
        for feat_idx, feat_name in enumerate(self.feature_names_):
            start = feat_idx * self.bits_per_feature_
            end = start + self.bits_per_feature_
            feat_bits = X_binary[:, start:end]
            densities[feat_name] = float(feat_bits.mean())
        return densities

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _bins_to_binary(self, bins: np.ndarray) -> np.ndarray:
        """Convert bin indices (n_samples, n_features) to binary (n_samples, n_features*bits)."""
        n_samples, n_features = bins.shape
        result = np.zeros((n_samples, n_features * self.bits_per_feature_), dtype=float)
        for feat_idx in range(n_features):
            for bit_idx in range(self.bits_per_feature_):
                col = feat_idx * self.bits_per_feature_ + bit_idx
                result[:, col] = (bins[:, feat_idx] >> (self.bits_per_feature_ - 1 - bit_idx)) & 1
        return result

    def _binary_to_bins(self, X_binary: np.ndarray) -> np.ndarray:
        """Convert binary (n_samples, n_features*bits) back to bin indices."""
        n_samples = X_binary.shape[0]
        result = np.zeros((n_samples, self.n_features_used_), dtype=int)
        for feat_idx in range(self.n_features_used_):
            for bit_idx in range(self.bits_per_feature_):
                col = feat_idx * self.bits_per_feature_ + bit_idx
                result[:, feat_idx] += (
                    X_binary[:, col].astype(int) << (self.bits_per_feature_ - 1 - bit_idx)
                )
        return result

    def _build_feature_map(self) -> Dict:
        """Build a map from qubit indices to feature names and bit positions."""
        feature_map = {}
        for feat_idx, feat_name in enumerate(self.feature_names_):
            start = feat_idx * self.bits_per_feature_
            end = start + self.bits_per_feature_
            for bit_idx, qubit_idx in enumerate(range(start, end)):
                feature_map[qubit_idx] = {
                    "feature": feat_name,
                    "bit_position": bit_idx,
                    "bits_per_feature": self.bits_per_feature_,
                    "method": self.binarize_method,
                }
        return feature_map

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """Convert input to DataFrame and select numeric columns."""
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        elif isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            raise TypeError(f"Expected DataFrame or ndarray, got {type(X)}")

        numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) == 0:
            raise ValueError("No numeric columns found in input data.")
        return X_df[numeric_cols]

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor is not fitted. Call fit() or fit_transform() first.")

    def summary(self) -> str:
        """Print a summary of the preprocessor configuration."""
        self._check_fitted()
        lines = [
            "FinancialDataPreprocessor Summary",
            "=" * 40,
            f"  Method          : {self.binarize_method}",
            f"  n_qubits        : {self.n_qubits}",
            f"  features used   : {self.n_features_used_}",
            f"  bits/feature    : {self.bits_per_feature_}",
            f"  total bits used : {self.n_features_used_ * self.bits_per_feature_}",
            f"  feature names   : {self.feature_names_}",
        ]
        return "\n".join(lines)
