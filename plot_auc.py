from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from utils import evaluate
import pandas as pd
from auc_compare import evaluate_boost_with_plots
import os

output_dir = 'output_results'
os.makedirs(output_dir, exist_ok=True)
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

"""ILPD"""
data = pd.read_csv('data/raw/Indian Liver Patient Dataset (ILPD).csv', header=None)
data.fillna(data.mean(), inplace=True)

# Encode
data.iloc[:, 1] = LabelEncoder().fit_transform(data.values[:, 1])

DATASETS.update({
    'ILPD': {
        'data': [data.values[:, :-1], data.values[:, -1]],
        'extra': {}
    }
})

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

for name, value in DATASETS.items():
    smh, hue, rus =  evaluate_boost_with_plots(
        "{} - Method: {}".format(name, ""),
        DecisionTreeClassifier(),
        *value.get('data'),
        **value.get('extra'),
        k=5,
        verbose=True
    )

    # Create a unique filename for each dataset
    output_filename = os.path.join(output_dir, f"{name}_results.txt")

    # Write the results to the file
    with open(output_filename, 'w') as f:
        f.write(f"Dataset: {name}\n\n")
        f.write(f"SMH Result:\n{smh}\n\n")
        f.write(f"HUE Result:\n{hue}\n\n")
        f.write(f"RUS Result:\n{rus}\n")
    
    print(f"Results saved to {output_filename}")
