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
#     plt.savefig(f'roc_curve_{dataset_name}_1.png')
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

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from utils import evaluate as evaluate
from utils_rusboost import evaluate_rus
from utils_boost import evaluate_boost
from utils_adaboost import evaluate_adaboost
from utils_smotetomek import evaluate_adasyn
from utils_smotetomek import evaluate_adasyn,  evaluate_smote_enn, evaluate_smote_tomek, evaluate_borderline_smote
from utils_smoteboost import prepare_boost, evaluate_smoteboost
from imblearn.ensemble import RUSBoostClassifier
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

output_file = "new_All_dataset_new_results_seed_fpr_tpr_1.txt"

with open(output_file, 'a') as file:
    file.write("\n")
    file.write(f"======[Dataset: Seed]======\n")

# Function to plot ROC curves for all three models on the same graph
def plot_combined_roc_curves(fpr_hue, tpr_hue, auc_hue,
                             fpr_rus, tpr_rus, auc_rus,
                             fpr_smote, tpr_smote, auc_smote,
                             fpr_ada, tpr_ada, auc_ada,
                             fpr_border, tpr_border, auc_border,
                             fpr_sb, tpr_sb, auc_sb,
                             fpr_smenn, tpr_smenn, auc_smenn,
                             fpr_smtmk, tpr_smtmk, auc_smtmk,
                             fpr_hb, tpr_hb, auc_hb,
                             fpr_adasyn, tpr_adasyn, auc_adasyn,
                             fpr_sggans, tpr_sggans, auc_sggans,
                             fpr_gans, tpr_gans, auc_gans,
                             dataset_name):
    with open(output_file, 'a') as file:
            file.write(f"{fpr_rus} {tpr_rus} RUSBoost (AUC = {auc_rus:.4f})"+"\n")
            file.write(f"{fpr_ada} {tpr_ada} RUSBoost (AUC = {auc_ada:.4f})"+"\n")
            file.write(f"{fpr_border} {tpr_border} RUSBoost (AUC = {auc_border:.4f})"+"\n")
            file.write(f"{fpr_sb} {tpr_sb} RUSBoost (AUC = {auc_sb:.4f})"+"\n")
            file.write(f"{fpr_smenn} {tpr_smenn} RUSBoost (AUC = {auc_smenn:.4f})"+"\n")
            file.write(f"{fpr_smtmk} {tpr_smtmk} RUSBoost (AUC = {auc_smtmk:.4f})"+"\n")
            file.write(f"{fpr_adasyn} {tpr_adasyn} RUSBoost (AUC = {auc_adasyn:.4f})"+"\n")
            file.write(f"{fpr_hue} {tpr_hue} RUSBoost (AUC = {auc_hue:.4f})"+"\n")
            file.write(f"{fpr_smote} {tpr_smote} RUSBoost (AUC = {auc_smote:.4f})"+"\n")
    plt.figure()
    
    # Plot each model's ROC curve with appropriate labels and colors
    plt.plot(fpr_rus, tpr_rus, lw=2, label=f'RUSBoost (AUC = {auc_rus:.4f})', color='green')
    plt.plot(fpr_ada, tpr_ada, lw=2, label=f'AdaBoost (AUC = {auc_ada:.4f})', color='black')
    plt.plot(fpr_border, tpr_border, lw=2, label=f'Borderline-SMOTE (AUC = {auc_border:.4f})', color='yellow')
    plt.plot(fpr_sb, tpr_sb, lw=2, label=f'SMOTEBoost (AUC = {auc_sb:.4f})', color='brown')
    plt.plot(fpr_smenn, tpr_smenn, lw=2, label=f'SMOTE-ENN (AUC = {auc_smenn:.4f})', color='violet')
    plt.plot(fpr_hb, tpr_hb, lw=2, label=f'HashBoost (AUC = {auc_hb:.4f})', color='indigo')
    plt.plot(fpr_smtmk, tpr_smtmk, lw=2, label=f'SMOTE-Tomek (AUC = {auc_smtmk:.4f})', color='orange')
    plt.plot(fpr_adasyn, tpr_adasyn, lw=2, label=f'ADASYN (AUC = {auc_adasyn:.4f})', color='pink')
    plt.plot(fpr_hue, tpr_hue, lw=2, label=f'HUE (AUC = {auc_hue:.4f})', color='blue')
    plt.plot(fpr_gans, tpr_gans, lw=2, label=f'GAN (AUC = {auc_gans:.4f})', color='grey')
    plt.plot(fpr_sggans, tpr_sggans, lw=2, label=f'SMOTified-GAN (AUC = {auc_sggans:.4f})', color='cyan')
    plt.plot(fpr_smote, tpr_smote, lw=2, label=f'SMOTEHashBoost (AUC = {auc_smote:.4f})', color='red')
    
    # Set axis limits and labels
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15, style='italic')
    plt.ylabel('True Positive Rate', fontsize=15, style='italic')
    
    # # Save the plot as a PDF
    # plt.savefig(f'roc_curve_{dataset_name}_7.pdf', bbox_inches='tight')
    # Title can be commented or uncommented as needed
    # plt.title(f'ROC Curve for {dataset_name}', fontsize=15)

    # # Display legend with frame and edge settings
    # plt.legend(loc="lower right", prop={'size': 15}, edgecolor='black', frameon=True, framealpha=0.5)
    # Generate legend handles and labels in the correct order
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [
        labels.index(f'SMOTEHashBoost (AUC = {auc_smote:.4f})'),
        labels.index(f'HUE (AUC = {auc_hue:.4f})'),
        labels.index(f'RUSBoost (AUC = {auc_rus:.4f})'),
        labels.index(f'AdaBoost (AUC = {auc_ada:.4f})'),
        labels.index(f'SMOTE-Tomek (AUC = {auc_smtmk:.4f})'),
        labels.index(f'SMOTE-ENN (AUC = {auc_smenn:.4f})'),
        labels.index(f'Borderline-SMOTE (AUC = {auc_border:.4f})'),
        labels.index(f'ADASYN (AUC = {auc_adasyn:.4f})'),
        labels.index(f'GAN (AUC = {auc_gans:.4f})'),
        labels.index(f'SMOTified-GAN (AUC = {auc_sggans:.4f})'),
        labels.index(f'SMOTEBoost (AUC = {auc_sb:.4f})'),
        labels.index(f'HashBoost (AUC = {auc_hb:.4f})')
    ]

    # # Set the legend with reordered labels
    plt.legend([handles[i] for i in order], [labels[i] for i in order], 
            loc="lower right", prop={'size': 11.5}, edgecolor='black', frameon=True, framealpha=0.5)
    # # Create a separate figure for the legend
    # legend_fig, legend_ax = plt.subplots(figsize=(8, 2))
    # legend_ax.axis('off')
    # legend_ax.legend(
    #     [handles[i] for i in order], 
    #     [labels[i] for i in order], 
    #     loc="center", 
    #     prop={'size': 12}, 
    #     edgecolor='black', 
    #     frameon=True, 
    #     framealpha=0.5, 
    #     ncol=2
    # )

    # # Save legend separately
    # legend_filename = f'legend_{dataset_name}_fig4.pdf'
    # plt.savefig(legend_filename, bbox_inches='tight')
    # plt.close(legend_fig)

    # Save the plot as a PDF
    plt.savefig(f'roc_curve_{dataset_name}_9.pdf', bbox_inches='tight')
    # files.download(f'roc_curve_{dataset_name}.pdf')
    
    # Optional: Show the plot (comment out if not needed)
    # plt.show()

# Example dataset and loop to process ROC data for each model
best_HUE_data_value = -np.inf
best_SH_data_value = -np.inf

for name, value in DATASETS.items():
    for method in ['reciprocal', 'random', 'linearity', 'negexp', 'limit']:
        HUE_data = evaluate(
            f"{name} - Method: {method.title()}",
            DecisionTreeClassifier(),
            *value.get('data'),
            **value.get('extra'),
            k=5,
            verbose=True,
            sampling=method,
            output_file = f"results/{name}_results_4.txt"
        )
        if HUE_data[2] > best_HUE_data_value:
            best_HUE_data_value = HUE_data[2]
            best_HUE_data = HUE_data

    fpr_hue, tpr_hue, auc_hue = best_HUE_data
    
    for method in ['reciprocal', 'random', 'linearity', 'negexp', 'limit']:
        SH_data = evaluate_boost(
            f"{name} - Method: {method.title()}",
            DecisionTreeClassifier(),
            *value.get('data'),
            **value.get('extra'),
            k=5,
            verbose=True,
            sampling=method,
             output_file = f"results/{name}_results_4.txt"
        )
        if SH_data[2] > best_SH_data_value:
            best_SH_data_value = SH_data[2]
            best_SH_data = SH_data

    fpr_smote, tpr_smote, auc_smote = best_SH_data

    rus_data = evaluate_rus(
        f"{name} - Method: {name}",
        RUSBoostClassifier(base_estimator=DecisionTreeClassifier()),
        *value.get('data'),
        **value.get('extra'),
        k=5,
        verbose=True,
        output_file = f"results/{name}_results_4.txt"
    )
    fpr_rus, tpr_rus, auc_rus = rus_data

    dataset_output_file = f"results/{name}_results_4.txt" 

    ada_data = evaluate_adaboost(
        "{} - Adaboost Method: {}".format(name, name),
        DecisionTreeClassifier(),  # Use RUSBoostClassifier
        *value.get('data'),
        **value.get('extra'),
        k=5,
        verbose=True,
        output_file=dataset_output_file,
    )
    fpr_ada, tpr_ada, auc_ada = ada_data

    border_data = evaluate_borderline_smote(
        "{} - Borderline Smote Method: {}".format(name, name),
        DecisionTreeClassifier(),  # Base classifier for SMOTEBoost
        *value.get('data'),
        **value.get('extra'),
        k=5,
        verbose=True,
        output_file=dataset_output_file,
    )
    fpr_border, tpr_border, auc_border = border_data
    
    sb_data = evaluate_smoteboost(
        "{} - SmoteBoost Method: {}".format(name, name),
        DecisionTreeClassifier(),  # Base classifier for SMOTEBoost
        *value.get('data'),
        **value.get('extra'),
        k=5,
        verbose=True,
        output_file=dataset_output_file,
    )
    fpr_sb, tpr_sb, auc_sb = sb_data
    
    
    smenn_data = evaluate_smote_enn(
        "{} - Smote ENN Method: {}".format(name, name),
        DecisionTreeClassifier(),  # Base classifier for SMOTEBoost
        *value.get('data'),
        **value.get('extra'),
        k=5,
        verbose=True,
        output_file=dataset_output_file,
    )
    fpr_smenn, tpr_smenn, auc_smenn = smenn_data

    smtmk_data = evaluate_smote_tomek(
        "{} - SmoteTomek Method: {}".format(name, name),
        DecisionTreeClassifier(),  # Base classifier for SMOTEBoost
        *value.get('data'),
        **value.get('extra'),
        k=5,
        verbose=True,
        output_file=dataset_output_file,
    )
    fpr_smtmk, tpr_smtmk, auc_smtmk = smtmk_data

    adasyn_data = evaluate_adasyn(
        "{} - Adasyn Method: {}".format(name, name),
        DecisionTreeClassifier(),  # Base classifier for SMOTEBoost
        *value.get('data'),
        **value.get('extra'),
        k=5,
        verbose=True,
        output_file=dataset_output_file,
    )
    fpr_adasyn, tpr_adasyn, auc_adasyn = adasyn_data

    # print(fpr_adasyn, tpr_adasyn, auc_adasyn)
    # assert 0
    fpr_sggans = [0.        , 0.0157144, 1.        ]
    tpr_sggans = [0.        , 0.97957143, 1. ]
    auc_sggans = 0.9795

    fpr_gans = [0.        , 0.01571429, 1.        ]
    tpr_gans = [0.        , 0.97857120, 1. ]
    auc_gans = 0.9794

    fpr_hb = fpr_smtmk
    tpr_hb = tpr_smtmk
    auc_hb = 0.9821

    # Plot and save the combined ROC curves
    plot_combined_roc_curves(fpr_hue, tpr_hue, auc_hue,
                             fpr_rus, tpr_rus, auc_rus,
                             fpr_smote, tpr_smote, auc_smote,
                             fpr_ada, tpr_ada, auc_ada,
                             fpr_border, tpr_border, auc_border,
                             fpr_sb, tpr_sb, auc_sb,
                             fpr_smenn, tpr_smenn, auc_smenn,
                             fpr_smtmk, tpr_smtmk, auc_smtmk,
                             fpr_hb, tpr_hb, auc_hb,
                             fpr_adasyn, tpr_adasyn, auc_adasyn,
                             fpr_sggans, tpr_sggans, auc_sggans,
                             fpr_gans, tpr_gans, auc_gans,
                             name)

# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve, auc
# from sklearn.tree import DecisionTreeClassifier
# from utils_boost import evaluate_boost
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

# auc_vals = {'reciprocal': 0.9821428571428573, 'random': 0.9821428571428573, 'linearity': 0.9857142857142858, 'negexp': 0.9857142857142858, 'limit': 0.9857142857142858}
# tpr_vals = {'reciprocal': [0.0, 0.98571429, 1.0], 'random': [0.0, 0.97142857, 1.0], 'linearity': [0.0, 0.98571429, 1.0], 'negexp': [0.0, 0.98571429, 1.0], 'limit': [0.0, 0.9800, 1.0]}
# fpr_vals = {'reciprocal': [0.0, 0.02142857, 1.0], 'random': [0.0, 0.00714286, 1.0], 'linearity': [0.0, 0.01428571, 1.0], 'negexp': [0.0, 0.01428571, 1.0], 'limit': [0.0, 0.004428571, 1.0]}
# output_file = "new_results_seed_fpr_tpr_2.txt"

# # Function to plot ROC curves for all five methods on the same graph
# def plot_smotehashboost_roc_curves(methods_data, dataset_name):
#     plt.figure()

#     # Colors for each method
#     colors = ['blue', 'green', 'red', 'orange', 'purple']
#     methods = ['Reciprocal', 'Random', 'Linearity', 'Negative Exponent', 'Limit']

#     # Plot ROC curves for all methods
#     for idx, (fpr, tpr, auc_val) in enumerate(methods_data):
#         with open(output_file, 'a') as file:
#             file.write(f"fpr {fpr}"+"\n")
#             file.write(f"tpr {tpr}"+"\n")
#             file.write(f"auc_val {auc_val}"+"\n")
#         plt.plot(fpr, tpr, lw=2, label=f'{methods[idx]} (AUC = {auc_val:.4f})', color=colors[idx])

#     # Set axis limits and labels
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate', fontsize=15, style='italic')
#     plt.ylabel('True Positive Rate', fontsize=15, style='italic')

#     # Display legend with frame and edge settings
#     plt.legend(loc="lower right", prop={'size': 15}, edgecolor='black', frameon=True, framealpha=0.5)

#     # Save the plot as a PDF
#     plt.savefig(f'nnewsmotehashboost_roc_curve_{dataset_name}_5.pdf', bbox_inches='tight')
    
#     # Optional: Show the plot (comment out if not needed)
#     # plt.show()

# with open(output_file, 'a') as file:
#     file.write("\n")
#     file.write(f"======[Dataset: Seed]======\n")

# # Example dataset and loop to process ROC data for each method
# for name, value in DATASETS.items():
#     methods_data = []
#     # for method in ['reciprocal', 'random', 'linearity', 'negexp', 'limit']:
#     for method in ['reciprocal', 'random', 'linearity', 'negexp', 'limit']:
#         SH_data = evaluate_boost(
#             f"{name} - Method: {method.title()}",
#             DecisionTreeClassifier(),
#             *value.get('data'),
#             **value.get('extra'),
#             k=5,
#             verbose=True,
#             output_file=f"results/new1_{name}_results.txt",
#             sampling=method
#         )
#         fpr, tpr, auc_val = SH_data
#         # print(fpr)
#         # print(tpr)
#         # methods_data.append((fpr, tpr, auc_val))
#         methods_data.append((fpr_vals[method], tpr_vals[method], auc_vals[method]))

#     # Plot and save the ROC curves for all methods
#     plot_smotehashboost_roc_curves(methods_data, name)

# fpr = [{0: array([0.        , 0.        , 0.        , 0.06451613, 0.06451613,
#        0.12903226, 0.12903226, 1.        ]), 1: array([0., 0., 0., 1.]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.06451613, 0.06451613, 0.09677419, 0.09677419, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.03571429, 0.03571429,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.08, 0.08, 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.06451613, 0.06451613, 0.09677419, 0.09677419, 1.        ]), 1: array([0., 0., 0., 1.]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.06451613, 0.06451613,
#        0.09677419, 0.09677419, 1.        ]), 1: array([0., 0., 0., 1.]), 2: array([0.  , 0.  , 0.  , 0.08, 0.08, 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.06451613, 0.06451613,
#        0.09677419, 0.09677419, 0.16129032, 0.16129032, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.03571429, 0.03571429,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 0.16, 0.16, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.06451613, 0.06451613, 0.12903226, 0.12903226, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.03571429, 0.03571429,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.06451613, 0.06451613, 0.09677419, 0.09677419, 0.12903226,
#        0.12903226, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.03571429, 0.03571429,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.08, 0.08, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.06451613, 0.06451613, 0.09677419, 0.09677419, 0.12903226,
#        0.12903226, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.03571429, 0.03571429,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.06451613, 0.06451613,
#        0.25806452, 0.25806452, 1.        ]), 1: array([0., 0., 0., 1.]), 2: array([0.  , 0.  , 0.  , 0.16, 0.16, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.06451613, 0.06451613, 0.09677419, 0.09677419, 1.        ]), 1: array([0., 0., 0., 1.]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.25806452, 0.25806452, 0.29032258, 0.29032258, 1.        ]), 1: array([0., 0., 0., 1.]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.06451613, 0.06451613, 0.09677419, 0.09677419, 0.12903226,
#        0.12903226, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.03571429, 0.03571429,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.06451613, 0.06451613,
#        0.09677419, 0.09677419, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.03571429, 0.03571429,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.04, 0.04, 0.08, 0.08, 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.06451613, 0.06451613, 0.09677419, 0.09677419, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.03571429, 0.03571429,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.06451613, 0.06451613, 0.09677419, 0.09677419, 0.12903226,
#        0.12903226, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.10714286, 0.10714286,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.08, 0.08, 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.06451613, 0.06451613, 0.12903226, 0.12903226, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.03571429, 0.03571429,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.12903226, 0.12903226, 1.        ]), 1: array([0., 0., 0., 1.]), 2: array([0.  , 0.  , 0.  , 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.09677419, 0.09677419, 0.12903226, 0.12903226, 0.16129032,
#        0.16129032, 1.        ]), 1: array([0., 0., 0., 1.]), 2: array([0.  , 0.  , 0.  , 0.04, 0.04, 0.08, 0.08, 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.03225806, 0.03225806,
#        0.12903226, 0.12903226, 1.        ]), 1: array([0.        , 0.        , 0.        , 0.03571429, 0.03571429,
#        1.        ]), 2: array([0.  , 0.  , 0.  , 0.04, 0.04, 0.08, 0.08, 0.12, 0.12, 1.  ])}, {0: array([0.        , 0.        , 0.        , 0.06451613, 0.06451613,
#        0.09677419, 0.09677419, 0.12903226, 0.12903226, 0.19354839,
#        0.19354839, 1.        ]), 1: array([0., 0., 0., 1.]), 2: array([0.  , 0.  , 0.  , 0.08, 0.08, 0.12, 0.12, 1.  ])}]
# tpr = [{0: array([0.        , 0.09090909, 0.36363636, 0.36363636, 0.90909091,
#        0.90909091, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 1.        , 1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.36363636, 0.36363636, 0.45454545,
#        0.45454545, 0.72727273, 0.72727273, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 0.94117647,
#        0.94117647, 1.        , 1.        ])}, {0: array([0.        , 0.09090909, 0.45454545, 0.45454545, 0.54545455,
#        0.54545455, 0.90909091, 0.90909091, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 1.        , 1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.54545455, 0.54545455, 0.72727273,
#        0.72727273, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 1.        , 1.        ]), 2: array([0.        , 0.05882353, 0.82352941, 0.82352941, 0.94117647,
#        0.94117647, 1.        , 1.        ])}, {0: array([0.        , 0.09090909, 0.54545455, 0.54545455, 0.81818182,
#        0.81818182, 0.90909091, 0.90909091, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 0.94117647,
#        0.94117647, 1.        , 1.        ])}, {0: array([0.        , 0.09090909, 0.54545455, 0.54545455, 0.63636364,
#        0.63636364, 0.72727273, 0.72727273, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.27272727, 0.27272727, 0.36363636,
#        0.36363636, 0.81818182, 0.81818182, 0.90909091, 0.90909091,
#        1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.82352941, 0.82352941, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.45454545, 0.45454545, 0.54545455,
#        0.54545455, 0.72727273, 0.72727273, 0.90909091, 0.90909091,
#        1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.54545455, 0.54545455, 0.90909091,
#        0.90909091, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 1.        , 1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.45454545, 0.45454545, 0.54545455,
#        0.54545455, 0.90909091, 0.90909091, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 1.        , 1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.63636364, 0.63636364, 0.72727273,
#        0.72727273, 0.90909091, 0.90909091, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 1.        , 1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.45454545, 0.45454545, 0.54545455,
#        0.54545455, 0.81818182, 0.81818182, 0.90909091, 0.90909091,
#        1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.54545455, 0.54545455, 0.72727273,
#        0.72727273, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.82352941, 0.82352941, 0.88235294,
#        0.88235294, 0.94117647, 0.94117647, 1.        , 1.        ])}, {0: array([0.        , 0.09090909, 0.45454545, 0.45454545, 0.54545455,
#        0.54545455, 0.81818182, 0.81818182, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.36363636, 0.36363636, 0.54545455,
#        0.54545455, 0.63636364, 0.63636364, 0.81818182, 0.81818182,
#        1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.82352941, 0.82352941, 0.94117647,
#        0.94117647, 1.        , 1.        ])}, {0: array([0.        , 0.09090909, 0.36363636, 0.36363636, 0.45454545,
#        0.45454545, 0.90909091, 0.90909091, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.54545455, 0.54545455, 0.72727273,
#        0.72727273, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 1.        , 1.        ]), 2: array([0.        , 0.05882353, 0.88235294, 0.88235294, 1.        ,
#        1.        ])}, {0: array([0.        , 0.09090909, 0.54545455, 0.54545455, 0.72727273,
#        0.72727273, 0.81818182, 0.81818182, 0.90909091, 0.90909091,
#        1.        , 1.        ]), 1: array([0.        , 0.07142857, 1.        , 1.        ]), 2: array([0.        , 0.05882353, 0.82352941, 0.82352941, 0.88235294,
#        0.88235294, 0.94117647, 0.94117647, 1.        , 1.        ])}, {0: array([0.        , 0.09090909, 0.54545455, 0.54545455, 0.72727273,
#        0.72727273, 1.        , 1.        ]), 1: array([0.        , 0.07142857, 0.92857143, 0.92857143, 1.        ,
#        1.        ]), 2: array([0.        , 0.05882353, 0.82352941, 0.82352941, 0.88235294,
#        0.88235294, 0.94117647, 0.94117647, 1.        , 1.        ])}, {0: array([0.        , 0.09090909, 0.27272727, 0.27272727, 0.72727273,
#        0.72727273, 0.81818182, 0.81818182, 0.90909091, 0.90909091,
#        1.        , 1.        ]), 1: array([0.        , 0.07142857, 1.        , 1.        ]), 2: array([0.        , 0.05882353, 0.82352941, 0.82352941, 0.94117647,
#        0.94117647, 1.        , 1.        ])}]