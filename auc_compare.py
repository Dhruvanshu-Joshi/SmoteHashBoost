from utils import prepare, evaluate
from utils_rusboost import prepare_rus
from utils_boost import prepare_boost
from sklearn.metrics import accuracy_score, roc_auc_score
from ensemble import HashBasedUndersamplingEnsemble
from imblearn.ensemble import RUSBoostClassifier 
from ensemble_boost import SmoteHashBoost
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_boost_with_plots(
        name,
        base_classifier,
        X,
        y,
        minority_class=None,
        k: int = 5,
        n_runs: int = 20,
        n_iterations: int = 50,
        random_state: int = None,
        verbose: bool = False,
        **kwargs
):
    """Model Evaluation

    :param name: str
        title of this classifier

    :param base_classifier:
        Base Classifier for Hashing-Based Undersampling Ensemble

    :param X: np.array (n_samples, n_features)
        Feature matrix

    :param y: np.array (n_samples,)
        labels vector

    :param minority_class: int or str (default = None)
        label of minority class
        if you want to set a specific class to be minority class

    :param k: int (default = 5)
        number of Folds (KFold)

    :param n_runs: int (default = 20)
        number of runs

    :param n_iterations: int (default = 50)
        number of iterations for Iterative Quantization of Hashing-Based Undersampling Ensemble

    :param random_state: int (default = None)
        seed of random generator

    :param verbose: bool (default = False)
        verbosity

    :return None
    """

    print()
    print("======[Dataset: {}]======".format(
        name
    ))

    np.random.seed(random_state)

    # Output template
    OUTPUT = "[{}] Accuracy: {:.4f}, AUC: {:.4f}"

    # Prepate the data (Make it Binary)
    X_sm, y_sm = prepare_boost(X, y, minority_class, verbose)

    # Prepate the data (Make it Binary)
    X_hu, y_hu = prepare_boost(X, y, minority_class, verbose)

    # Prepate the data (Make it Binary)
    X_ru, y_ru = prepare_boost(X, y, minority_class, verbose)

    folds = np.zeros((n_runs, 2))
    rus=[]
    hue=[]
    smotehash=[]
    for run in tqdm(range(n_runs)):

        # Applying k-Fold (k = 5 due to the paper)
        kFold = StratifiedKFold(n_splits=k, shuffle=True)

        # store metrics in this variable
        metrics = np.zeros((k, 2))
        for fold, (trIndexes, tsIndexes) in enumerate(kFold.split(X, y)):
            # Split data to Train and Test
            Xtr_sm, ytr_sm = X_sm[trIndexes], y_sm[trIndexes]
            Xts_sm, yts_sm = X_sm[tsIndexes], y_sm[tsIndexes]

            Xtr_hu, ytr_hu = X_hu[trIndexes], y_hu[trIndexes]
            Xts_hu, yts_hu = X_hu[tsIndexes], y_hu[tsIndexes]

            Xtr_ru, ytr_ru = X_ru[trIndexes], y_ru[trIndexes]
            Xts_ru, yts_ru = X_ru[tsIndexes], y_ru[tsIndexes]

            # Define Model
            model_ru = RUSBoostClassifier(
                base_estimator=base_classifier,
                random_state=random_state,
                **kwargs
            )

            AUC_hu = 0

            for method in [
                'reciprocal',
                'random',
                'linearity',
                'negexp',
                'limit'
            ]:
                # Define Model
                model_hu = HashBasedUndersamplingEnsemble(
                    base_estimator=base_classifier,
                    n_iterations=n_iterations,
                    random_state=random_state,
                    sampling=method,
                    **kwargs
                )

                # Fit the training data on the model
                model_hu.fit(Xtr_hu, ytr_hu)

                # Predict the test data
                predicted_hu = model_hu.predict(Xts_hu)

                # AUC evaluation
                auc_hu = roc_auc_score(yts_hu, predicted_hu)
                AUC_hu = max(auc_hu, AUC_hu)
            hue.append(AUC_hu)

            AUC_sm = 0

            for method in [
                'reciprocal',
                'random',
                'linearity',
                'negexp',
                'limit'
            ]:
                # Define Model
                model_sm = SmoteHashBoost(
                    base_estimator=base_classifier,
                    n_iterations=n_iterations,
                    random_state=random_state,
                    **kwargs
                )

                # Fit the training data on the model
                model_sm.fit(Xtr_sm, ytr_sm)

                # Predict the test data
                predicted_sm = model_sm.predict(Xts_sm)

                # AUC evaluation
                auc_sm = roc_auc_score(yts_sm, predicted_sm)
                AUC_sm = max(auc_sm, AUC_sm)

            smotehash.append(AUC_sm)

            # Fit the training data on the model
            model_ru.fit(Xtr_ru, ytr_ru)

            # Predict the test data
            predicted_ru = model_ru.predict(Xts_ru)

            # AUC evaluation
            AUC_ru = roc_auc_score(yts_ru, predicted_ru)

            rus.append(AUC_ru)

    best_auc_sm = max(smotehash)
    best_auc_hu = max(hue)
    best_auc_ru = max(rus)