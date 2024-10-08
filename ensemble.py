from sklearn.decomposition import PCA
from scipy.linalg import svd
from copy import deepcopy
from typing import List
import numpy as np

# np.warnings_.filterwarnings('ignore')


class HashBasedUndersamplingEnsemble:
    RECIPROCAL, RANDOM, LINEARITY, NEGATIVE_EXPONENT, LIMIT = [
        'reciprocal',
        'random',
        'linearity',
        'negexp',
        'limit'
    ]

    SUPPORTED_SMPLINGS = [
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
            random_state: int = None
    ):
        """
        Hashing-Based Undersampling Ensemble for Imbalanced Pattern Classification Problems

        :param base_estimator:
            Base Estimator

        :param sampling: str (default = 'normal')
            sampling method
            supported methods: 'reciprocal', 'normal', 'random', 'linearity', 'negexp', 'limit'

        :param n_iterations: int (default = 50)
            maximum iteration for Iterative Quantization

        :param random_state: int (default = None)
            random state for Iterative Quantization

        """
        self.base_estimator = base_estimator

        self.sampling: str = sampling
        if self.sampling not in self.SUPPORTED_SMPLINGS:
            raise ValueError('supported sampling: {}'.format(
                self.SUPPORTED_SMPLINGS
            ))

        self.n_iterations: int = n_iterations
        if type(self.n_iterations) != int \
                or not (0 < self.n_iterations):
            raise ValueError('n_iterations should be an integer number bigger than 0')

        self.random_state: int = random_state
        np.random.seed(self.random_state)

        # store classifiers
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
        """Sign

        Apply Sign function over X

        :param X: np.array
            Input

        :return np.array
            Sign(X)
        """
        return np.where(X >= 0, 1, -1)

    def _itq(self, X: np.array):
        """Iterative Quantitization

        :param X: np.array (n_sample, n_features)
            projected feature matrix via PCA

        :return R: np.array
            rotate matrix
        """

        # Construct Orthogonal rotation matrix
        R = np.random.randn(self.n_bits, self.n_bits)
        [U, _, _] = svd(R)
        R = U[:, :self.n_bits]

        # Find Optimal Rotation
        for _ in range(self.n_iterations):
            V = X @ R
            [U, _, VT] = svd(self._sign(V).T @ X)
            R = (VT @ U.T)

        return R

    def _sampling(self, X: np.array, subspace: np.array):
        """Sampling Methods
        1. Reciprocal
        2. All Random
        3. Linearity
        4. Negative Exponent
        5. Limit
        """
        # get number of samples
        n_samples, _ = X.shape

        # Calculate Hamming Distance for all sample
        distance = np.sum(
            np.unpackbits(X ^ subspace, axis=1, count=self.n_bits, bitorder='little')
            , axis=1
        )

        if self.sampling == self.RANDOM:
            """All Random"""
            return np.random.choice(n_samples, self._nMin)

        elif self.sampling == self.LINEARITY:
            """Linearity"""

            # calculate weights
            weights = (self.n_bits + 1 - distance) / (self.n_bits + 1)

        elif self.sampling == self.NEGATIVE_EXPONENT:
            """Negative Exponent"""

            # calculate weights
            weights = 1 / (np.power(2, distance))

        elif self.sampling == self.LIMIT:
            """Limit"""

            # calculate weights
            weights = np.where(distance == 0, 1, 0)

        else:
            """Reciprocal"""

            # calculate weights
            weights = np.nan_to_num(
                1 / (distance * np.power(2, self.n_bits))
                , nan=1, neginf=1, posinf=1
            )

        # Shuffle weights for sampling (we can make the sampling randomness
        # for selecting from the surronding subspaces)
        np.random.shuffle(weights)

        # Sort weights by their weights and so Pick Nmin samples due to weight
        # distribution to form the training subset
        return np.argsort(weights)[::-1][:self._nMin]

    def fit(self, X: np.array, y: np.array):
        """Fitting Function

        X: np.array (n_samples, n_features)
            features matrix

        y: np.array (n_samples,)
            labels vector
        """

        # Validate X and y
        X, y = self._check_Xy(X, y)

        # Get number of Bits
        # (we need to handle this value for PCA projection conditions, like n_components for solvers)
        self.n_bits = np.min([
            np.ceil(np.log2(3 * self._nMaj / self._nMin)).astype(np.int),
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

            # Pick Nmin samples due to weight distribution w to form the training subset
            selected = self._sampling(Q, subspace)

            # Prepare training data for classifier
            X_ = np.concatenate((X[self._minIndexes], self._majX[selected]))
            y_ = np.concatenate((y[self._minIndexes], self._majY[selected]))

            # Train base classifier C using T and minority samples
            C = deepcopy(self.base_estimator)

            # store all classifiers for prediction step
            self.classifiers.append(C.fit(X_, y_))

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
        # FIXME: what about other labels ? what if the labels be something else 1 and -1 ?
        return self._sign(H)
