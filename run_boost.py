from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from utils_boost import prepare_boost
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score,roc_curve
from ensemble import HashBasedUndersamplingEnsemble
from ensemble_vg import SMOTEHashBasedEnsemble
from ensemble_boost import SmoteHashBoost
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from utils import prepare
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from ensemble import HashBasedUndersamplingEnsemble
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np

DATASETS = dict()

"""Wine Dataset"""
X, y = load_wine(return_X_y=True)
DATASETS.update({
    'Wine': {
        'data': [X, y],
        'extra': {
        }
    }
})

"""Flare-F"""
data = pd.read_csv('data/raw/flare-F.dat', header=None)
objects = data.select_dtypes(include=['object'])
for col in objects.columns:
    if col == len(data.columns) - 1:
        continue
    data.iloc[:, col] = LabelEncoder().fit_transform(data.values[:, col])

DATASETS.update({
    'Flare-F': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {

        }
    }
})

"""Yeast5"""
data = pd.read_csv('data/raw/yeast5.dat', header=None)
DATASETS.update({
    'Yeast5': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {}
    }
})

"""Car vGood"""
data = pd.read_csv('data/raw/car.data', header=None)
DATASETS.update({
    'CarvGood': {
        'data': [
            OrdinalEncoder().fit_transform(data.values[:, :-1]),
            data.values[:, -1]
        ],
        'extra': {
            'minority_class': 'vgood'
        }
    }
})


"""Car Good"""
data = pd.read_csv('data/raw/car.data', header=None)
DATASETS.update({
    'CarGood': {
        'data': [
            OrdinalEncoder().fit_transform(data.values[:, :-1]),
            data.values[:, -1]
        ],
        'extra': {
            'minority_class': 'good'
        }
    }
})

"""Seed"""
data = pd.read_csv('data/raw/seeds_dataset.txt', header=None)
DATASETS.update({
    'Seed': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {
            'minority_class': 2
        }
    }
})

"""Glass"""
data = pd.read_csv('data/raw/glass.csv', header=None)
DATASETS.update({
    'Glass': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {
            'minority_class': 7
        }
    }
})

# """ILPD"""
# data = pd.read_csv('data/raw/Indian Liver Patient Dataset (ILPD).csv', header=None)
# data.fillna(data.mean(), inplace=True)

# Encode
# data.iloc[:, 1] = LabelEncoder().fit_transform(data.values[:, 1])

# DATASETS.update({
#     'ILPD': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {}
#     }
# })

"""Yeast5-ERL"""
data = pd.read_csv('data/raw/yeast5.data', header=None)
DATASETS.update({
    'Yeast5-ERL': {
        'data': [data.values[:, 1:-1], data.values[:, -1]],
        'extra': {
            # 'minority_class': 'ME1'
            'minority_class': 'ERL'
        }
    }
})


def evaluate_boost(
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
    """Model Evaluation for SmoteHashBoost with ROC curve plotting capabilities."""

    print(f"======[Dataset: {name}]======")

    np.random.seed(random_state)

    # Output template
    OUTPUT = "[{}] Accuracy: {:.4f}, AUC: {:.4f}"

    # Prepare the data (Make it Binary)
    X, y = prepare_boost(X, y, minority_class, verbose)

    # List to store ROC curve data
    roc_data = []

    for run in tqdm(range(n_runs)):

        # Applying k-Fold cross-validation (Stratified K-Fold)
        kFold = StratifiedKFold(n_splits=k, shuffle=True)

        for fold, (trIndexes, tsIndexes) in enumerate(kFold.split(X, y)):
            # Split data into Train and Test
            Xtr, ytr = X[trIndexes], y[trIndexes]
            Xts, yts = X[tsIndexes], y[tsIndexes]

            # Define the SmoteHashBoost model
            model = SmoteHashBoost(
                base_estimator=base_classifier,
                n_iterations=n_iterations,
                random_state=random_state,
                **kwargs
            )

            # Fit the training data on the model
            model.fit(Xtr, ytr)

            # Predict probabilities if available, otherwise use the class predictions
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(Xts)[:, 1]  # Probability for positive class
            else:
                y_prob = model.predict(Xts)  # Use predicted class if `predict_proba` is not available

            # AUC evaluation
            auc_score = roc_auc_score(yts, y_prob)

            # Accuracy evaluation
            predicted = model.predict(Xts)
            accuracy = accuracy_score(yts, predicted)

            # Collect ROC curve data
            fpr, tpr, _ = roc_curve(yts, y_prob)
            roc_data.append((fpr, tpr, auc_score))

    print(OUTPUT.format("Best", accuracy, auc_score))

    # Return ROC data for plotting
    return roc_data
# for name, value in DATASETS.items():
#     for method in [
#         'reciprocal',
#         'random',
#         'linearity',
#         'negexp',
#         'limit'
#     ]:
#         evaluate_boost(
#             "{} - Method: {}".format(name, method.title()),
#             DecisionTreeClassifier(),
#             *value.get('data'),
#             **value.get('extra'),
#             k=5,
#             verbose=True,
#             sampling=method
#         )
#     print("*"*50)
