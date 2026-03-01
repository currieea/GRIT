import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC


class Direct_Invariant_L2_Kernel(BaseEstimator, ClassifierMixin):
    def __init__(self, gamma=0.5, invariance_strength=1.0, C=1.0, normalize_pairs=True):
        """
        normalize_pairs : bool
            If True, the invariance penalty is averaged over the number of pairs.
            This allows 'invariance_strength' to remain stable regardless of
            how many pairs are provided.
        """
        self.gamma = gamma
        self.invariance_strength = invariance_strength
        self.C = C
        self.normalize_pairs = normalize_pairs

        if invariance_strength > 1e-9:
            self.base_lambda = 1.0 / (2.0 * invariance_strength)
        else:
            self.base_lambda = np.inf

        self.X_train = None
        self.Z = None
        self.Z_prime = None
        self.classifier = None
        self.B_train = None
        self.cho_factors = None
        self._inv_solve_cache = None

    def _get_diff_kernel_matrices(self, X, Z, Z_prime):
        K_xz  = rbf_kernel(X, Z,       gamma=self.gamma)
        K_xz_p = rbf_kernel(X, Z_prime, gamma=self.gamma)
        return K_xz - K_xz_p

    def _get_delta_delta_kernel(self, Z, Z_prime):
        K_zz   = rbf_kernel(Z,       Z,       gamma=self.gamma)
        K_zz_p  = rbf_kernel(Z,       Z_prime, gamma=self.gamma)
        K_zp_z  = rbf_kernel(Z_prime, Z,       gamma=self.gamma)
        K_zp_zp = rbf_kernel(Z_prime, Z_prime, gamma=self.gamma)
        return K_zz - K_zz_p - K_zp_z + K_zp_zp

    def fit(self, X, y, Z, Z_prime):
        self.X_train = X
        self.Z       = Z
        self.Z_prime = Z_prime

        K_xx = rbf_kernel(X, X, gamma=self.gamma)

        if self.base_lambda == np.inf:
            self.classifier = SVC(kernel="precomputed", C=self.C)
            self.classifier.fit(K_xx, y)
            return self

        K_pairs = Z.shape[0]
        lambda_eff = (
            self.base_lambda * K_pairs
            if self.normalize_pairs and K_pairs > 0
            else self.base_lambda
        )

        self.B_train = self._get_diff_kernel_matrices(X, Z, Z_prime)   # (N, K)
        C_mat        = self._get_delta_delta_kernel(Z, Z_prime)         # (K, K)

        matrix_to_invert = C_mat.copy()
        idx = np.diag_indices_from(matrix_to_invert)
        matrix_to_invert[idx] += lambda_eff + 1e-8

        self.cho_factors = cho_factor(matrix_to_invert, lower=True)

        # Cache (M^-1 B^T) for fast prediction
        self._inv_solve_cache = cho_solve(self.cho_factors, self.B_train.T)
        correction   = self.B_train @ self._inv_solve_cache
        K_invariant  = K_xx - correction

        self.classifier = SVC(kernel="precomputed", C=self.C)
        self.classifier.fit(K_invariant, y)
        return self

    def predict(self, X):
        K_test_train = rbf_kernel(X, self.X_train, gamma=self.gamma)

        if self.base_lambda == np.inf:
            return self.classifier.predict(K_test_train)

        B_test           = self._get_diff_kernel_matrices(X, self.Z, self.Z_prime)
        correction        = B_test @ self._inv_solve_cache   # reuse cached solve
        K_invariant_test  = K_test_train - correction
        return self.classifier.predict(K_invariant_test)
