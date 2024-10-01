from sklearn.decomposition import PCA
from scipy.linalg import svd
from copy import deepcopy
from typing import List
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from imblearn.over_sampling import SMOTE

class SmoteHashBoost:
    RECIPROCAL, RANDOM, LINEARITY, NEGATIVE_EXPONENT, LIMIT = [
        'reciprocal',
        'random',
        'linearity',
        'negexp',
        'limit'
    ]

    SUPPORTED_SAMPLINGS = [
        RECIPROCAL,
        RANDOM,
        LINEARITY,
        NEGATIVE_EXPONENT,
        LIMIT
    ]

    def __init__(
            self,
            base_estimator,
            sampling: str = RECIPROCAL,
            n_iterations: int = 50,
            k_neighbors: int = 5,
            n_estimators: int = 50,
            random_state: int = None
    ):
        """
        SmoteHashBoost Ensemble for Imbalanced Pattern Classification Problems

        :param base_estimator:
            Base Estimator

        :param sampling: str (default = 'reciprocal')
            sampling method
            supported methods: 'reciprocal', 'random', 'linearity', 'negexp', 'limit'

        :param n_iterations: int (default = 50)
            maximum iteration for Iterative Quantization

        :param k_neighbors: int (default = 5)
            Number of nearest neighbors to use in SMOTE

        :param n_estimators: int (default = 50)
            Number of boosting rounds

        :param random_state: int (default = None)
            random state for reproducibility
        """
        self.base_estimator = base_estimator
        self.sampling: str = sampling
        self.n_iterations: int = n_iterations
        self.k_neighbors: int = k_neighbors
        self.n_estimators: int = n_estimators
        self.random_state: int = random_state

        np.random.seed(self.random_state)

        if self.sampling not in self.SUPPORTED_SAMPLINGS:
            raise ValueError('supported sampling methods: {}'.format(
                self.SUPPORTED_SAMPLINGS
            ))

        if type(self.n_iterations) != int or not (0 < self.n_iterations):
            raise ValueError('n_iterations should be an integer number bigger than 0')

        # Boosting classifier
        self.boosting = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=self.n_estimators, random_state=self.random_state)

        # Store classifiers
        self.classifiers: List = list()

    def _check_fitted(self):
        assert self._is_fitted, 'fit function not called yet'

    def _check_Xy(self, X, y: np.array = None) -> [np.array, np.array]:
        """Check X and y to be valid"""
        if len(X.shape) != 2:
            raise ValueError('X should be 2D (n_samples x n_features)')

        if y is not None:
            n_samples, n_features = X.shape
            if len(y.flatten()) != n_samples:
                raise ValueError('number of samples in y is not equal to X')

            self.classes_, self.n_classes_ = np.unique(y, return_counts=True)
            if len(self.classes_) > 2:
                raise NotImplementedError('Just binary class supported'
                                          ', multi class not supported yet')

            # Get indexes of sorted number of each class
            sorted_indexes = np.argsort(self.n_classes_)

            # Label of each class
            self.minC, self.majC = self.classes_[sorted_indexes]

            # Number of each class
            self._nMin, self._nMaj = self.n_classes_[sorted_indexes]

            # get indexes of minority and majority classes
            self._minIndexes = np.where(y != self.majC)[0]
            self._majIndexes = np.where(y == self.majC)[0]

            # separate X and Y of majority class from whole data
            self._majX, self._majY = X[self._majIndexes], y[self._majIndexes]

        return X, y

    def _sign(self, X: np.array) -> np.array:
        """Sign Function: Apply Sign function over X"""
        return np.where(X >= 0, 1, -1)

    def _itq(self, X: np.array):
        """Iterative Quantization"""
        R = np.random.randn(self.n_bits, self.n_bits)
        [U, _, _] = svd(R)
        R = U[:, :self.n_bits]

        for _ in range(self.n_iterations):
            V = X @ R
            [U, _, VT] = svd(self._sign(V).T @ X)
            R = (VT @ U.T)

        return R

    def _sampling(self, X: np.array, subspace: np.array):
        """Sampling Methods"""
        n_samples, _ = X.shape
        distance = np.sum(
            np.unpackbits(X ^ subspace, axis=1, count=self.n_bits, bitorder='little'),
            axis=1
        )

        if self.sampling == self.RANDOM:
            return np.random.choice(n_samples, self._nMin)

        elif self.sampling == self.LINEARITY:
            weights = (self.n_bits + 1 - distance) / (self.n_bits + 1)

        elif self.sampling == self.NEGATIVE_EXPONENT:
            weights = 1 / (np.power(2, distance))

        elif self.sampling == self.LIMIT:
            weights = np.where(distance == 0, 1, 0)

        else:
            weights = np.nan_to_num(
                1 / (distance * np.power(2, self.n_bits)),
                nan=1, neginf=1, posinf=1
            )

        np.random.shuffle(weights)
        return np.argsort(weights)[::-1][:self._nMin]

    def fit(self, X: np.array, y: np.array):
        """Fitting Function"""
        X, y = self._check_Xy(X, y)

        # Dynamically adjust k_neighbors to avoid "Expected n_neighbors <= n_samples" error
        minority_count = len(self._minIndexes)
        k_neighbors = min(self.k_neighbors, minority_count - 1) if minority_count > 1 else 1

        # Apply SMOTE oversampling
        smote = SMOTE(k_neighbors=k_neighbors, random_state=self.random_state)
        X_res, y_res = smote.fit_resample(X, y)

        # Re-check X and y after resampling
        X_res, y_res = self._check_Xy(X_res, y_res)

        # Get number of Bits
        self.n_bits = np.min([
            np.ceil(np.log2(3 * self._nMaj / self._nMin)).astype(int),
            *X.shape
        ])

        # Using PCA (Dimension Reduction)
        self.pca = PCA(n_components=self.n_bits)

        # Transform X
        V = self.pca.fit_transform(self._majX)

        # Using Iterative Quantitization (Rotation Matrix)
        self.R = self._itq(V)

        # V x R
        U = self._sign(V @ self.R).astype(np.int)

        # Assign each sample to Hash Code Subspace
        Q = np.packbits(np.where(U < 0, 0, U), axis=1, bitorder='little')

        for subspace in range(np.power(2, self.n_bits)):
            selected = self._sampling(Q, subspace)
            X_ = np.concatenate((X_res[self._minIndexes], self._majX[selected]))
            y_ = np.concatenate((y_res[self._minIndexes], self._majY[selected]))

            # Train base classifier C using T and minority samples
            C = deepcopy(self.base_estimator)

            # Store all classifiers for prediction step
            self.classifiers.append(C.fit(X_, y_))

        # Boosting Step
        self.boosting.fit(X_res, y_res)
        self._is_fitted = True
        return self

    def predict(self, X: np.array):
        """Prediction Function"""
        self._check_fitted()

        # Check and normalize X
        X, _ = self._check_Xy(X)

        # Prediction step
        H = np.sum([
            classifier.predict(X) for classifier in self.classifiers
        ], axis=0)

        # Apply sign function over result of classifiers
        return self._sign(H)
