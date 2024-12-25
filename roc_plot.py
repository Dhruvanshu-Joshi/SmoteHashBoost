# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# from sklearn.tree import DecisionTreeClassifier
# # Import the evaluation methods from the respective files
# from utils import evaluate as evaluate
# from utils_rusboost import evaluate_rus
# from utils_boost import evaluate_boost  # Assuming SmoteHashBoost uses evaluate_boost
# from sklearn.datasets import load_wine
# import pandas as pd
# from tqdm import tqdm
# import numpy as np
# from imblearn.ensemble import RUSBoostClassifier
# from ensemble_boost import SmoteHashBoost
# from ensemble import HashBasedUndersamplingEnsemble
# import matplotlib

# DATASETS = dict()

# # """Wine Dataset"""
# # X, y = load_wine(return_X_y=True)
# # DATASETS.update({
# #     'Wine': {
# #         'data': [X, y],
# #         'extra': {
# #         }
# #     }
# # })

# # """Flare-F"""
# # data = pd.read_csv('data/raw/flare-F.dat', header=None)
# # objects = data.select_dtypes(include=['object'])
# # for col in objects.columns:
# #     if col == len(data.columns) - 1:
# #         continue
# #     data.iloc[:, col] = LabelEncoder().fit_transform(data.values[:, col])

# # DATASETS.update({
# #     'Flare-F': {
# #         'data': [data.values[:, :-1], data.values[:, -1]],
# #         'extra': {

# #         }
# #     }
# # })

# # """Yeast5"""
# # data = pd.read_csv('data/raw/yeast5.dat', header=None)
# # DATASETS.update({
# #     'Yeast5': {
# #         'data': [data.values[:, :-1], data.values[:, -1]],
# #         'extra': {}
# #     }
# # })

# # """Car vGood"""
# # data = pd.read_csv('data/raw/car.data', header=None)
# # DATASETS.update({
# #     'CarvGood': {
# #         'data': [
# #             OrdinalEncoder().fit_transform(data.values[:, :-1]),
# #             data.values[:, -1]
# #         ],
# #         'extra': {
# #             'minority_class': 'vgood'
# #         }
# #     }
# # })


# # """Car Good"""
# # data = pd.read_csv('data/raw/car.data', header=None)
# # DATASETS.update({
# #     'CarGood': {
# #         'data': [
# #             OrdinalEncoder().fit_transform(data.values[:, :-1]),
# #             data.values[:, -1]
# #         ],
# #         'extra': {
# #             'minority_class': 'good'
# #         }
# #     }
# # })

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

# # """Glass"""
# # data = pd.read_csv('data/raw/glass.csv', header=None)
# # DATASETS.update({
# #     'Glass': {
# #         'data': [data.values[:, :-1], data.values[:, -1]],
# #         'extra': {
# #             'minority_class': 7
# #         }
# #     }
# # })

# # """ILPD"""
# # data = pd.read_csv('data/raw/Indian Liver Patient Dataset (ILPD).csv', header=None)
# # data.fillna(data.mean(), inplace=True)

# # # Encode
# # data.iloc[:, 1] = LabelEncoder().fit_transform(data.values[:, 1])

# # DATASETS.update({
# #     'ILPD': {
# #         'data': [data.values[:, :-1], data.values[:, -1]],
# #         'extra': {}
# #     }
# # })

# # """Yeast5-ERL"""
# # data = pd.read_csv('data/raw/yeast5.data', header=None)
# # DATASETS.update({
# #     'Yeast5-ERL': {
# #         'data': [data.values[:, 1:-1], data.values[:, -1]],
# #         'extra': {
# #             # 'minority_class': 'ME1'
# #             'minority_class': 'ERL'
# #         }
# #     }
# # })


# # # Assuming you have datasets loaded
# # datasets = [
# #  "Yeast5-ERL"
# # ]
# # "Wine", "Flare-F", "Yeast5", "CarvGood", "CarGood", 
# #     "Seed", "Glass", "Yeast5-ERL"

# # Function to plot ROC curves for all three models on the same graph
# def plot_combined_roc_curves(fpr_hue, tpr_hue, auc_hue,
#                              fpr_rus, tpr_rus, auc_rus,
#                              fpr_smote, tpr_smote, auc_smote,
#                              dataset_name):
#     plt.figure()
#     plt.plot(fpr_hue, tpr_hue, lw=2, label=f'HUE (AUC = {auc_hue:.4f})', color='blue')
#     plt.plot(fpr_rus, tpr_rus, lw=2, label=f'RusBoost (AUC = {auc_rus:.4f})', color='green')
#     plt.plot(fpr_smote, tpr_smote, lw=2, label=f'SmoteHashBoost (AUC = {auc_smote:.4f})', color='red')
    
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title(f'ROC Curve for {dataset_name}')
#     plt.legend(loc="lower right")

#     # Save the plot as an image
#     plt.savefig(f'roc_curve_{dataset_name}.png')
#     plt.show()

# best_HUE_data = None
# best_HUE_data_value = -np.inf
# best_SH_data = None
# best_SH_data_value = -np.inf
# for name, value in DATASETS.items():
#     for method in [
#         'reciprocal',
#         'random',
#         'linearity',
#         'negexp',
#         'limit'
#     ]:
#         HUE_data = evaluate(
#             "{} - Method: {}".format(name, method.title()),
#             DecisionTreeClassifier(),
#             *value.get('data'),
#             **value.get('extra'),
#             k=5,
#             verbose=True,
#             sampling=method
#         )
#         # Check if the current HUE_data has a higher value at index 2
#         if HUE_data[2] > best_HUE_data_value:
#             best_HUE_data_value = HUE_data[2]
#             best_HUE_data = HUE_data  # Update to the current best data
#     fpr_hue, tpr_hue, auc_hue = HUE_data
#     print("*"*50)
#     for method in [
#         'reciprocal',
#         'random',
#         'linearity',
#         'negexp',
#         'limit'
#     ]:
#         SH_data = evaluate_boost(
#             "{} - Method: {}".format(name, method.title()),
#             DecisionTreeClassifier(),
#             *value.get('data'),
#             **value.get('extra'),
#             k=5,
#             verbose=True,
#             sampling=method
#         )
#         # Check if the current HUE_data has a higher value at index 2
#         if SH_data[2] > best_SH_data_value:
#             best_SH_data_value = SH_data[2]
#             best_SH_data = SH_data  # Update to the current best data
#     fpr_smote, tpr_smote, auc_smote = SH_data
#     print("*"*50)
#     rus_data = evaluate_rus(
#         "{} - Method: {}".format(name, name),
#         RUSBoostClassifier(base_estimator=DecisionTreeClassifier()),  # Use RUSBoostClassifier
#         *value.get('data'),
#         **value.get('extra'),
#         k=5,
#         verbose=True,
#     )
#     fpr_rus, tpr_rus, auc_rus = rus_data
#     print("*"*50)

#     plot_combined_roc_curves(fpr_hue, tpr_hue, auc_hue,
#                              fpr_rus, tpr_rus, auc_rus,
#                              fpr_smote, tpr_smote, auc_smote,
#                              name)

#*****************************

# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.tree import DecisionTreeClassifier
# from utils import evaluate as evaluate
# from utils_rusboost import evaluate_rus
# from utils_boost import evaluate_boost
# from imblearn.ensemble import RUSBoostClassifier
# import pandas as pd
# import numpy as np
# import matplotlib

# # Set matplotlib parameters for font and label sizes
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rc('xtick', labelsize=15)
# matplotlib.rc('ytick', labelsize=15)

# DATASETS = dict()

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

# # Function to plot ROC curves for all three models on the same graph
# def plot_combined_roc_curves(fpr_hue, tpr_hue, auc_hue,
#                              fpr_rus, tpr_rus, auc_rus,
#                              fpr_smote, tpr_smote, auc_smote,
#                              dataset_name):
#     plt.figure()
    
#     # Plot each model's ROC curve with appropriate labels and colors
#     plt.plot(fpr_hue, tpr_hue, lw=2, label=f'HUE (AUC = {auc_hue:.2f})', color='blue')
#     plt.plot(fpr_rus, tpr_rus, lw=2, label=f'RUSBoost (AUC = {auc_rus:.2f})', color='green')
#     plt.plot(fpr_smote, tpr_smote, lw=2, label=f'SmoteHashBoost (AUC = {auc_smote:.2f})', color='red')
    
#     # Set axis limits and labels
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate', fontsize=15, style='italic')
#     plt.ylabel('True Positive Rate', fontsize=15, style='italic')
    
#     # Title can be commented or uncommented as needed
#     # plt.title(f'ROC Curve for {dataset_name}', fontsize=15)

#     # Display legend with frame and edge settings
#     plt.legend(loc="lower right", prop={'size': 15}, edgecolor='black', frameon=True, framealpha=0.5)

#     # Save the plot as a PDF
#     plt.savefig(f'roc_curve_{dataset_name}.pdf', bbox_inches='tight')
#     # files.download(f'roc_curve_{dataset_name}.pdf')
    
#     # Optional: Show the plot (comment out if not needed)
#     # plt.show()

# # Example dataset and loop to process ROC data for each model
# best_HUE_data_value = -np.inf
# best_SH_data_value = -np.inf
# for name, value in DATASETS.items():
#     for method in ['reciprocal', 'random', 'linearity', 'negexp', 'limit']:
#         HUE_data = evaluate(
#             f"{name} - Method: {method.title()}",
#             DecisionTreeClassifier(),
#             *value.get('data'),
#             **value.get('extra'),
#             k=5,
#             verbose=True,
#             sampling=method
#         )
#         if HUE_data[2] > best_HUE_data_value:
#             best_HUE_data_value = HUE_data[2]
#             best_HUE_data = HUE_data

#     fpr_hue, tpr_hue, auc_hue = best_HUE_data
    
#     for method in ['reciprocal', 'random', 'linearity', 'negexp', 'limit']:
#         SH_data = evaluate_boost(
#             f"{name} - Method: {method.title()}",
#             DecisionTreeClassifier(),
#             *value.get('data'),
#             **value.get('extra'),
#             k=5,
#             verbose=True,
#             sampling=method
#         )
#         if SH_data[2] > best_SH_data_value:
#             best_SH_data_value = SH_data[2]
#             best_SH_data = SH_data

#     fpr_smote, tpr_smote, auc_smote = best_SH_data

#     rus_data = evaluate_rus(
#         f"{name} - Method: {name}",
#         RUSBoostClassifier(base_estimator=DecisionTreeClassifier()),
#         *value.get('data'),
#         **value.get('extra'),
#         k=5,
#         verbose=True,
#     )
#     fpr_rus, tpr_rus, auc_rus = rus_data

#     # Plot and save the combined ROC curves
#     plot_combined_roc_curves(fpr_hue, tpr_hue, auc_hue,
#                              fpr_rus, tpr_rus, auc_rus,
#                              fpr_smote, tpr_smote, auc_smote,
#                              name)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from utils_boost import evaluate_boost
import pandas as pd
import numpy as np
import matplotlib

# Set matplotlib parameters for font and label sizes
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

DATASETS = dict()

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

# Function to plot ROC curves for all five methods on the same graph
def plot_smotehashboost_roc_curves(methods_data, dataset_name):
    plt.figure()

    # Colors for each method
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    methods = ['Reciprocal', 'Random', 'Linearity', 'Negexp', 'Limit']

    # Plot ROC curves for all methods
    for idx, (fpr, tpr, auc_val) in enumerate(methods_data):
        plt.plot(fpr, tpr, lw=2, label=f'{methods[idx]} (AUC = {auc_val:.4f})', color=colors[idx])

    # Set axis limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15, style='italic')
    plt.ylabel('True Positive Rate', fontsize=15, style='italic')

    # Display legend with frame and edge settings
    plt.legend(loc="lower right", prop={'size': 15}, edgecolor='black', frameon=True, framealpha=0.5)

    # Save the plot as a PDF
    plt.savefig(f'smotehashboost_roc_curve_{dataset_name}.pdf', bbox_inches='tight')
    
    # Optional: Show the plot (comment out if not needed)
    # plt.show()

# Example dataset and loop to process ROC data for each method
for name, value in DATASETS.items():
    methods_data = []
    for method in ['reciprocal', 'random', 'linearity', 'negexp', 'limit']:
        SH_data = evaluate_boost(
            f"{name} - Method: {method.title()}",
            DecisionTreeClassifier(),
            *value.get('data'),
            **value.get('extra'),
            k=5,
            verbose=True,
            sampling=method
        )
        fpr, tpr, auc_val = SH_data
        methods_data.append((fpr, tpr, auc_val))

    # Plot and save the ROC curves for all methods
    plot_smotehashboost_roc_curves(methods_data, name)
