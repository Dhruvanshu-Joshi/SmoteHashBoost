import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
# Import the evaluation methods from the respective files
from run import evaluate as evaluate_hue
from run_rusboost import evaluate_rus
from run_boost import evaluate_boost  # Assuming SmoteHashBoost uses evaluate_boost
from sklearn.datasets import load_wine
import pandas as pd
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


# Assuming you have datasets loaded
datasets = [
 "Yeast5-ERL"
]
# "Wine", "Flare-F", "Yeast5", "CarvGood", "CarGood", 
#     "Seed", "Glass", "Yeast5-ERL"

# Function to plot ROC curves for all three models on the same graph
def plot_combined_roc_curves(fpr_hue, tpr_hue, auc_hue,
                             fpr_rus, tpr_rus, auc_rus,
                             fpr_smote, tpr_smote, auc_smote,
                             dataset_name):
    plt.figure()
    plt.plot(fpr_hue, tpr_hue, lw=2, label=f'HUE (AUC = {auc_hue:.2f})', color='blue')
    plt.plot(fpr_rus, tpr_rus, lw=2, label=f'RusBoost (AUC = {auc_rus:.2f})', color='green')
    plt.plot(fpr_smote, tpr_smote, lw=2, label=f'SmoteHashBoost (AUC = {auc_smote:.2f})', color='red')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {dataset_name}')
    plt.legend(loc="lower right")

    # Save the plot as an image
    plt.savefig(f'roc_curve_{dataset_name}.png')
    plt.show()

# Loop through all datasets and plot ROC curves for all models
for dataset in datasets:
    
    # Run HUE model and get ROC data
    roc_data_hue = evaluate_hue(dataset, DecisionTreeClassifier(), *DATASETS[dataset]['data'])
    fpr_hue, tpr_hue, auc_hue = roc_data_hue[0]
    
    # Run RusBoost model and get ROC data
    roc_data_rusboost = evaluate_rus(dataset, DecisionTreeClassifier(), *DATASETS[dataset]['data'])
    fpr_rus, tpr_rus, auc_rus = roc_data_rusboost[0]
    
    # Run SmoteHashBoost model and get ROC data
    roc_data_smotehashboost = evaluate_boost(dataset, DecisionTreeClassifier(), *DATASETS[dataset]['data'])
    fpr_smote, tpr_smote, auc_smote = roc_data_smotehashboost[0]

    # Plot combined ROC curves for HUE, RusBoost, and SmoteHashBoost for the current dataset
    plot_combined_roc_curves(fpr_hue, tpr_hue, auc_hue,
                             fpr_rus, tpr_rus, auc_rus,
                             fpr_smote, tpr_smote, auc_smote,
                             dataset)
