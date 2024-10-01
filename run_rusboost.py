from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from imblearn.ensemble import RUSBoostClassifier  # Import RUSBoost from imbalanced-learn library
from utils_rusboost import prepare_rus, evaluate_rus
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from imblearn.ensemble import RUSBoostClassifier  #
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

# """Wine Dataset"""
# X, y = load_wine(return_X_y=True)
# DATASETS.update({
#     'Wine': {
#         'data': [X, y],
#         'extra': {
#         }
#     }
# })

# """Flare-F"""
# data = pd.read_csv('data/raw/flare-F.dat', header=None)
# objects = data.select_dtypes(include=['object'])
# for col in objects.columns:
#     if col == len(data.columns) - 1:
#         continue
#     data.iloc[:, col] = LabelEncoder().fit_transform(data.values[:, col])

# DATASETS.update({
#     'Flare-F': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {

#         }
#     }
# })

# """Yeast5"""
# data = pd.read_csv('data/raw/yeast5.dat', header=None)
# DATASETS.update({
#     'Yeast5': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {}
#     }
# })

# """Car vGood"""
# data = pd.read_csv('data/raw/car.data', header=None)
# DATASETS.update({
#     'CarvGood': {
#         'data': [
#             OrdinalEncoder().fit_transform(data.values[:, :-1]),
#             data.values[:, -1]
#         ],
#         'extra': {
#             'minority_class': 'vgood'
#         }
#     }
# })


# """Car Good"""
# data = pd.read_csv('data/raw/car.data', header=None)
# DATASETS.update({
#     'CarGood': {
#         'data': [
#             OrdinalEncoder().fit_transform(data.values[:, :-1]),
#             data.values[:, -1]
#         ],
#         'extra': {
#             'minority_class': 'good'
#         }
#     }
# })

# """Seed"""
# data = pd.read_csv('data/raw/seeds_dataset.txt', header=None)
# DATASETS.update({
#     'Seed': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {
#             'minority_class': 2
#         }
#     }
# })

# """Glass"""
# data = pd.read_csv('data/raw/glass.csv', header=None)
# DATASETS.update({
#     'Glass': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {
#             'minority_class': '7'
#         }
#     }
# })

# # """ILPD"""
# # data = pd.read_csv('data/raw/Indian Liver Patient Dataset (ILPD).csv', header=None)
# # data.fillna(data.mean(), inplace=True)

# # Encode
# # data.iloc[:, 1] = LabelEncoder().fit_transform(data.values[:, 1])

# # DATASETS.update({
# #     'ILPD': {
# #         'data': [data.values[:, :-1], data.values[:, -1]],
# #         'extra': {}
# #     }
# # })

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

# # """Skin"""
# # data = pd.read_csv('data/raw/Skin_NonSkin.txt', delimiter='\t', header=None)
# # DATASETS.update({
# #     'Skin': {
# #         'data': [data.values[:, :-1], data.values[:, -1]],
# #         'extra': {}
# #     }
# # })

# # """Letter Dataset"""
# # data = pd.read_csv('data/raw/letter-recognition.data', header=None)
# # DATASETS.update({
# #     'Letter': {
# #         'data': [data.values[:, 1:], data.values[:, 0]],
# #         'extra': {
# #             'minority_class': 'A'
# #         }
# #     }
# # })

# # """MNIST"""
# # data = pd.read_csv('data/raw/mnist_784.csv', header=0)
# # DATASETS.update({
# #     'DIGITS': {
# #         'data': [data.values[:, :-1], data.values[:, -1]],
# #         'extra': {
# #             'minority_class': 4,
# #             'n_runs': 1
# #         }
# #     }
# # })


def evaluate_rus_1(
        name,
        base_classifier,
        X,
        y,
        minority_class=None,
        k: int = 5,
        n_runs: int = 20,
        random_state: int = None,
        verbose: bool = False,
        **kwargs
):
    """Model Evaluation with ROC curve plotting capabilities for RUSBoost

    :param name: str
        title of this classifier

    :param base_classifier:
        Base Classifier for RUSBoost

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

    :param random_state: int (default = None)
        seed of random generator

    :param verbose: bool (default = False)
        verbosity

    :return List of ROC data (fpr, tpr, auc)
    """

    print()
    print("======[Dataset: {}]======".format(name))

    np.random.seed(random_state)

    # Output template
    OUTPUT = "[{}] Accuracy: {:.4f}, AUC: {:.4f}"

    # Prepare the data (Make it Binary)
    X, y = prepare_rus(X, y, minority_class, verbose)

    # List to store ROC curve data (fpr, tpr, auc)
    roc_data = []

    for run in tqdm(range(n_runs)):

        # Applying k-Fold cross-validation (Stratified K-Fold)
        kFold = StratifiedKFold(n_splits=k, shuffle=True)

        # Store metrics in this variable
        for fold, (trIndexes, tsIndexes) in enumerate(kFold.split(X, y)):
            # Split data into training and test sets
            Xtr, ytr = X[trIndexes], y[trIndexes]
            Xts, yts = X[tsIndexes], y[tsIndexes]

            # Define the RUSBoost model
            model = RUSBoostClassifier(
                base_estimator=base_classifier,
                random_state=random_state,
                **kwargs
            )
            # Fit the training data on the model
            model.fit(Xtr, ytr)

            # Predict probabilities (for ROC curve)
            y_prob = model.predict_proba(Xts)[:, 1]  # Get probability for positive class

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


for name, value in DATASETS.items():
    # for method in [
    #     'reciprocal',
    #     'random',
    #     'linearity',
    #     'negexp',
    #     'limit'
    # ]:
    evaluate_rus(
        "{} - Method: {}".format(name, name),
        RUSBoostClassifier(base_estimator=DecisionTreeClassifier()),  # Use RUSBoostClassifier
        *value.get('data'),
        **value.get('extra'),
        k=5,
        verbose=True,
    )
    print("*"*50)
