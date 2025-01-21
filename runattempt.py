from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from imblearn.ensemble import RUSBoostClassifier  # Import RUSBoost from imbalanced-learn library
from utils import prepare, evaluate
from utils_adaboost import prepare_adaboost, evaluate_adaboost
from utils_smotetomek import evaluate_adasyn,  evaluate_smote_enn, evaluate_smote_tomek, evaluate_borderline_smote
from utils_boost import prepare_boost, evaluate_boost
from utils_rusboost import prepare_rus, evaluate_rus
from utils_smoteboost import prepare_boost, evaluate_smoteboost
from ensemble_boost import SmoteHashBoost
from ensemble import HashBasedUndersamplingEnsemble
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.datasets import load_wine
from utils import prepare
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
#             'minority_class': 7
#         }
#     }
# })

# """ILPD"""
# data = pd.read_csv('data/raw/Indian Liver Patient Dataset (ILPD).csv', header=None)
# data.fillna(data.mean(), inplace=True)

# #Encode
# data.iloc[:, 1] = LabelEncoder().fit_transform(data.values[:, 1])

# DATASETS.update({
#     'ILPD': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {}
#     }
# })

# """Yeast5-ERL"""
# data = pd.read_csv('data/raw/yeast5.data', header=None)
# DATASETS.update({
#     'Yeast5-ERL': {
#         'data': [data.values[:, 1:-1], data.values[:, -1]],
#         'extra': {
#             # 'minority_class': 'ME1'
#             'minority_class': 'ERL'
#         }
#     }
# })

# data = pd.read_csv('data/raw/higgs.csv', header=0)

# # Assuming the last column contains labels 's' and 'b'
# label_encoder = LabelEncoder()
# data.iloc[:, -1] = label_encoder.fit_transform(data.iloc[:, -1])  # Encode 's' -> 1, 'b' -> 0

# print(data)

# # Split into features and target
# X = data.values[:, :-1]  # Features
# y = data.values[:, -1]   # Encoded target (0 for 'b', 1 for 's')

# # Update the DATASETS dictionary
# DATASETS.update({
#     'HIGGS': {
#         'data': [X, y],
#         'extra': {
#         }
#     }
# })

# data = pd.read_csv('data/raw/higgs.csv', header=0)
# print(data.values[:, -1])
# DATASETS.update({
#     'HIGGS': {
#         'data': [
#             OrdinalEncoder().fit_transform(data.values[:, :-1]),
#             data.values[:, -1]
#         ],
#         'extra': {
#             'minority_class': 's'
#         }
#     }
# })


# # Load the KDD Cup 1999 dataset
# data = pd.read_csv('data/raw/kdd_cup_new.csv', header=0)

# # # Assuming the last column contains the categorical labels (e.g., dos, normal, probe, r2l, u2r)
# # label_encoder = LabelEncoder()
# # data.iloc[:, -1] = label_encoder.fit_transform(data.iloc[:, -1])  # Encode categorical labels

# # # Split the dataset into features and target
# # X = OrdinalEncoder().fit_transform(data.values[:, :-1])  # Features
# # y = data.values[:, -1]   # Encoded target

# # print(data)

# # print(X)
# # print(y)

# # # Count the occurrences of each label in the dataset
# # unique_labels, counts = pd.Series(data.iloc[:, -1]).value_counts().index, pd.Series(data.iloc[:, -1]).value_counts().values
# # label_distribution = {label_encoder.inverse_transform([label])[0]: count for label, count in zip(unique_labels, counts)}

# # # Update the DATASETS dictionary
# # DATASETS.update({
# #     'KDD Cup 1999': {
# #         'data': [X, y],
# #         'extra': {
# #         }
# #     }
# # })
# DATASETS.update({
#     'kdd_cup_new': {
#         'data': [data.values[:, :-1], data.values[:, -1]],
#         'extra': {
#             # 'minority_class': 'ME1'
#             # 'minority_class': 'ERL'
#         }
#     }
# })


# # Load the Epileptic Seizure Recognition dataset
# data = pd.read_csv('data/raw/seizure.csv', header=0, low_memory=False)

# # data.iloc[:, -1] = data.iloc[:, -1].astype(str)

# # # Assuming the last column contains the categorical labels (e.g., Seizure and Non-seizure)
# # label_encoder = LabelEncoder()
# # data.iloc[:, -1] = label_encoder.fit_transform(data.iloc[:, -1])  # Encode 'Seizure' -> 1, 'Non-seizure' -> 0

# # # Split the dataset into features and target
# # X = data.values[:, :-1]  # Features
# # y = data.values[:, -1]   # Encoded target

# # # Count occurrences of each label
# # label_distribution = {
# #     class_label: (y == class_index).sum()
# #     for class_index, class_label in enumerate(label_encoder.classes_)
# # }

# # # Update the DATASETS dictionary
# # DATASETS.update({
# #     'Epileptic Seizure Recognition': {
# #         'data': [X, y],
# #         'extra': {
# #         }
# #     }
# # })

# DATASETS.update({
#     'Epileptic Seizure Recognition': {
#         'data': [OrdinalEncoder().fit_transform(data.values[:, :-1]), data.values[:, -1]],
#         'extra': {}
#     }
# })


# # Load the breast cancer dataset
# data = pd.read_csv('data/raw/breast_cancer.csv', header=None)

# # Encode categorical features if necessary
# objects = data.select_dtypes(include=['object'])
# for col in objects.columns:
#     if col == data.shape[1] - 1:  # Skip the last column if it's the target
#         continue
#     data.iloc[:, col] = LabelEncoder().fit_transform(data.iloc[:, col])

# # Update the DATASETS dictionary
# DATASETS.update({
#     'Breast Cancer Wisconsin': {
#         'data': [data.iloc[:, :-1].values, data.iloc[:, -1].values],  # Features and target
#         'extra': {

#         }
#     }
# })


'''Diabetes'''
data = pd.read_csv('data/raw/diabetes_data.csv', header=0)

data.fillna(data.mean(), inplace=True)

DATASETS.update({
    'Diabetes': {
        'data': [OrdinalEncoder().fit_transform(data.values[:, :-1]), data.values[:, -1]],
        'extra': {}
    }
})


# '''sonar'''
# data = pd.read_csv('data/raw/sonar_all_data.csv', header=None)

# DATASETS.update({
#     'Sonar': {
#         'data': [OrdinalEncoder().fit_transform(data.values[:, :-1]), data.values[:, -1]],
#         'extra': {}
#     }
# })


# '''student_dropout'''
# data = pd.read_csv('data/raw/student_dropout.csv', header=0)

# DATASETS.update({
#     'Sonar': {
#         'data': [OrdinalEncoder().fit_transform(data.values[:, :-1]), data.values[:, -1]],
#         'extra': {}
#     }
# })


# '''default of credit card clients'''
# data = pd.read_excel('data/raw/default of credit card clients.xls', header=0)

# DATASETS.update({
#     'default of credit card clients': {
#         'data': [OrdinalEncoder().fit_transform(data.values[:, :-1]), data.values[:, -1]],
#         'extra': {}
#     }
# })


for name, value in DATASETS.items():
    dataset_output_file = f"results/{name}_resultextra.txt"
    # evaluate_adaboost(
    #     "{} - Adaboost Method: {}".format(name, name),
    #     DecisionTreeClassifier(),  # Use RUSBoostClassifier
    #     *value.get('data'),
    #     **value.get('extra'),
    #     k=5,
    #     verbose=True,
    #     output_file=dataset_output_file,
    # )
    # print("*"*50)
    evaluate_rus(
        "{} - Rusboost Method: {}".format(name, name),
        RUSBoostClassifier(base_estimator=DecisionTreeClassifier()),  # Use RUSBoostClassifier
        *value.get('data'),
        **value.get('extra'),
        k=5,
        verbose=True,
        output_file=dataset_output_file,
    )
    print("*"*50)
    # for method in [
    #     'reciprocal',
    #     'random',
    #     'linearity',
    #     'negexp',
    #     'limit'
    # ]:
    #     evaluate_boost(
    #         "{} - SmoteHashBoost Method: {}".format(name, method.title()),
    #         DecisionTreeClassifier(),
    #         *value.get('data'),
    #         **value.get('extra'),
    #         k=5,
    #         verbose=True,
    #         output_file=dataset_output_file,
    #         sampling=method
    #     )
    # print("*"*50)
    # for method in [
    #     'reciprocal',
    #     'random',
    #     'linearity',
    #     'negexp',
    #     'limit'
    # ]:
    #     evaluate(
    #         "{} - Method: {}".format(name, method.title()),
    #         DecisionTreeClassifier(),
    #         *value.get('data'),
    #         **value.get('extra'),
    #         k=5,
    #         verbose=True,
    #         sampling=method
    #     )
    # print("*"*50)
    # evaluate_borderline_smote(
    #     "{} - Borderline Smote Method: {}".format(name, name),
    #     DecisionTreeClassifier(),  # Base classifier for SMOTEBoost
    #     *value.get('data'),
    #     **value.get('extra'),
    #     k=5,
    #     verbose=True,
    #     output_file=dataset_output_file,
    # )
    # print("*" * 50)
    # evaluate_smoteboost(
    #     "{} - SmoteBoost Method: {}".format(name, name),
    #     DecisionTreeClassifier(),  # Base classifier for SMOTEBoost
    #     *value.get('data'),
    #     **value.get('extra'),
    #     k=5,
    #     verbose=True,
    #     output_file=dataset_output_file,
    # )
    # print("*" * 50)
    # evaluate_smote_enn(
    #     "{} - Smote ENN Method: {}".format(name, name),
    #     DecisionTreeClassifier(),  # Base classifier for SMOTEBoost
    #     *value.get('data'),
    #     **value.get('extra'),
    #     k=5,
    #     verbose=True,
    #     output_file=dataset_output_file,
    # )
    # evaluate_smote_tomek(
    #     "{} - SmoteTomek Method: {}".format(name, name),
    #     DecisionTreeClassifier(),  # Base classifier for SMOTEBoost
    #     *value.get('data'),
    #     **value.get('extra'),
    #     k=5,
    #     verbose=True,
    #     output_file=dataset_output_file,
    # )
    # try:
    #     evaluate_adasyn(
    #         "{} - Adasyn Method: {}".format(name, name),
    #         DecisionTreeClassifier(),  # Base classifier for SMOTEBoost
    #         *value.get('data'),
    #         **value.get('extra'),
    #         k=5,
    #         verbose=True,
    #         output_file=dataset_output_file,
    #     )
    # except RuntimeError as e:
    #     print(f"[ADASYN Failed] {e}")
    # print("*" * 50)
