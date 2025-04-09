'''working code 1'''
# from sklearn.decomposition import PCA
# from scipy.linalg import svd
# from copy import deepcopy
# from typing import List
# import numpy as np
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.cluster import KMeans
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import SMOTE

# class CSRBoosting:
#     def __init__(
#             self,
#             base_estimator,
#             n_clusters: int = None,
#             n_estimators: int = 50,
#             k_neighbors: int = 5,
#             random_state: int = None,
#             cluster_percent_of_minority: float = 100.0
#     ):
#         """
#         Clustered Sampling with Resampling Boosting (CSRBoosting) for Imbalanced Classification
        
#         :param base_estimator: Base classifier for boosting
#         :param n_clusters: Number of clusters for majority class (default: equal to minority class size)
#         :param n_estimators: Number of estimators for boosting
#         :param k_neighbors: Number of neighbors for SMOTE
#         :param random_state: Random seed for reproducibility
#         """
#         self.base_estimator = base_estimator
#         self.n_clusters = n_clusters
#         self.n_estimators = n_estimators
#         self.k_neighbors = k_neighbors
#         self.random_state = random_state
#         self.boosting = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=random_state)
#         self.classifiers: List = []
    
#     def _check_Xy(self, X, y):
#         if len(X.shape) != 2:
#             raise ValueError('X should be 2D (n_samples x n_features)')
        
#         unique_classes, class_counts = np.unique(y, return_counts=True)
#         if len(unique_classes) > 2:
#             raise NotImplementedError('Only binary classification is supported')
        
#         min_class, maj_class = unique_classes[np.argsort(class_counts)]
#         min_idx, maj_idx = np.where(y == min_class)[0], np.where(y == maj_class)[0]
#         return X, y, min_class, maj_class, min_idx, maj_idx
    
#     def fit(self, X, y):
#         X, y, min_class, maj_class, min_idx, maj_idx = self._check_Xy(X, y)
        
#         # Clustering the majority class
#         n_clusters = self.n_clusters if self.n_clusters else len(min_idx)
#         kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
#         cluster_labels = kmeans.fit_predict(X[maj_idx])
        
#         # Apply random undersampling within each cluster
#         new_majority = []
#         for cluster in np.unique(cluster_labels):
#             cluster_indices = maj_idx[cluster_labels == cluster]
#             sample_size = max(1, len(cluster_indices) // 2)  # Reduce size per cluster
#             rng = np.random.default_rng(self.random_state)
#             sampled_indices = rng.choice(cluster_indices, sample_size, replace=False)
#             new_majority.extend(sampled_indices)
#         new_majority = np.array(new_majority)
        
#         # Define target size for minority class
#         target_minority_size = len(new_majority)
        
#         # Apply SMOTE to increase minority samples
#         # smote = SMOTE(sampling_strategy=target_minority_size / len(min_idx), k_neighbors=min(self.k_neighbors, len(min_idx)-1), random_state=self.random_state)
#         smote = SMOTE(
#             sampling_strategy={min_class: target_minority_size},
#             k_neighbors=min(self.k_neighbors, len(min_idx)-1),
#             random_state=self.random_state
#         )
#         X_res, y_res = smote.fit_resample(X, y)
        
#         # Train boosting classifier
#         self.boosting.fit(X_res, y_res)
#         self._is_fitted = True
#         return self
    
#     def predict(self, X):
#         if not hasattr(self, '_is_fitted'):
#             raise ValueError('Model is not fitted yet')
#         return self.boosting.predict(X)

'''code with percent but few small errors'''
# class CSRBoosting:
#     def __init__(
#             self,
#             base_estimator,
#             cluster_percent_of_minority: float = 100.0,  # new: percentage as float (e.g., 200 means 2x the minority count)
#             n_estimators: int = 50,
#             k_neighbors: int = 5,
#             random_state: int = None
#     ):
#         """
#         Clustered Sampling with Resampling Boosting (CSRBoosting) for Imbalanced Classification

#         :param base_estimator: Base classifier for boosting
#         :param cluster_percent_of_minority: Percentage multiplier (x) to compute number of clusters as (x / 100) * minority_class_size
#         :param n_estimators: Number of estimators for boosting
#         :param k_neighbors: Number of neighbors for SMOTE
#         :param random_state: Random seed for reproducibility
#         """
#         self.base_estimator = base_estimator
#         self.cluster_percent_of_minority = cluster_percent_of_minority
#         self.n_estimators = n_estimators
#         self.k_neighbors = k_neighbors
#         self.random_state = random_state
#         self.boosting = AdaBoostClassifier(
#             base_estimator=base_estimator,
#             n_estimators=n_estimators,
#             random_state=random_state
#         )
#         self.classifiers: List = []

#     def _check_Xy(self, X, y):
#         if len(X.shape) != 2:
#             raise ValueError('X should be 2D (n_samples x n_features)')
        
#         unique_classes, class_counts = np.unique(y, return_counts=True)
#         if len(unique_classes) > 2:
#             raise NotImplementedError('Only binary classification is supported')

#         min_class, maj_class = unique_classes[np.argsort(class_counts)]
#         min_idx, maj_idx = np.where(y == min_class)[0], np.where(y == maj_class)[0]
#         return X, y, min_class, maj_class, min_idx, maj_idx

#     def fit(self, X, y):
#         X, y, min_class, maj_class, min_idx, maj_idx = self._check_Xy(X, y)

#         # ✅ Number of clusters = (x / 100) * minority size
#         n_clusters = int((self.cluster_percent_of_minority / 100.0) * len(min_idx))
#         n_clusters = max(1, min(n_clusters, len(maj_idx)))  # ensure valid number of clusters

#         # Clustering the majority class
#         kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
#         cluster_labels = kmeans.fit_predict(X[maj_idx])

#         # Undersample majority within each cluster
#         new_majority = []
#         for cluster in np.unique(cluster_labels):
#             cluster_indices = maj_idx[cluster_labels == cluster]
#             sample_size = max(1, len(cluster_indices) // 2)
#             sampled_indices = np.random.choice(cluster_indices, sample_size, replace=False)
#             new_majority.extend(sampled_indices)
#         new_majority = np.array(new_majority)

#         # Target minority size = new_majority size
#         target_minority_size = len(new_majority)

#         try:
#             if target_minority_size <= len(min_idx):
#                 raise ValueError(f"Requested {target_minority_size} minority samples, but only {len(min_idx)} available.")
#             # Apply SMOTE
#             smote = SMOTE(
#                 sampling_strategy={min_class: target_minority_size},
#                 k_neighbors=min(self.k_neighbors, len(min_idx) - 1),
#                 random_state=self.random_state
#             )
#             X_res, y_res = smote.fit_resample(X, y)
#         except ValueError as e:
#             print("minority class less:", str(e))
#             self._is_fitted = False
#             return self  # skip training and allow graceful exit

#         # Train
#         self.boosting.fit(X_res, y_res)
#         self._is_fitted = True
#         return self

#     def predict(self, X):
#         if not hasattr(self, '_is_fitted'):
#             raise ValueError('Model is not fitted yet')
#         return self.boosting.predict(X)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import numpy as np
from typing import List

class CSRBoosting:
    def __init__(
            self,
            base_estimator,
            n_clusters: int = None,
            n_estimators: int = 50,
            k_neighbors: int = 5,
            random_state: int = None,
            cluster_percent_of_minority: float = 100.0
    ):
        """
        Clustered Sampling with Resampling Boosting (CSRBoosting) for Imbalanced Classification

        :param base_estimator: Base classifier for boosting
        :param n_clusters: Number of clusters for majority class (default: equal to minority class size)
        :param n_estimators: Number of estimators for boosting
        :param k_neighbors: Number of neighbors for SMOTE
        :param random_state: Random seed for reproducibility
        :param cluster_percent_of_minority: % of minority class to use for defining #clusters
        """
        self.base_estimator = base_estimator
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.cluster_percent_of_minority = cluster_percent_of_minority
        self.boosting = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, random_state=random_state)
        self.classifiers: List = []

    def _check_Xy(self, X, y):
        if len(X.shape) != 2:
            raise ValueError('X should be 2D (n_samples x n_features)')

        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) > 2:
            raise NotImplementedError('Only binary classification is supported')

        min_class, maj_class = unique_classes[np.argsort(class_counts)]
        min_idx, maj_idx = np.where(y == min_class)[0], np.where(y == maj_class)[0]
        return X, y, min_class, maj_class, min_idx, maj_idx

    def fit(self, X, y):
        X, y, min_class, maj_class, min_idx, maj_idx = self._check_Xy(X, y)

        # Decide number of clusters
        target_clusters = self.n_clusters if self.n_clusters else int((self.cluster_percent_of_minority / 100.0) * len(min_idx))
        target_clusters = max(1, target_clusters)

        # Cluster majority class
        kmeans = KMeans(n_clusters=target_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X[maj_idx])

        # Apply RUS within each cluster
        rng = np.random.default_rng(self.random_state)
        new_majority = []
        for cluster in np.unique(cluster_labels):
            cluster_indices = maj_idx[cluster_labels == cluster]
            sample_size = max(1, len(cluster_indices) // 2)
            if sample_size > len(cluster_indices):
                sample_size = len(cluster_indices)
            sampled_indices = rng.choice(cluster_indices, sample_size, replace=False)
            new_majority.extend(sampled_indices)
        new_majority = np.array(new_majority)

        # Combine with minority class
        combined_indices = np.concatenate([min_idx, new_majority])
        X_comb, y_comb = X[combined_indices], y[combined_indices]

        # Re-identify majority/minority after RUS
        _, _, new_min_class, new_maj_class, new_min_idx, _ = self._check_Xy(X_comb, y_comb)
        target_minority_size = np.sum(y_comb == new_maj_class)

        # Apply SMOTE on new minority class
        smote = SMOTE(
            sampling_strategy={new_min_class: target_minority_size},
            k_neighbors=min(self.k_neighbors, len(new_min_idx) - 1),
            random_state=self.random_state
        )
        X_res, y_res = smote.fit_resample(X_comb, y_comb)

        # Train boosting model
        self.boosting.fit(X_res, y_res)
        self._is_fitted = True
        return self

    def predict(self, X):
        if not hasattr(self, '_is_fitted'):
            raise ValueError('Model is not fitted yet')
        return self.boosting.predict(X)
