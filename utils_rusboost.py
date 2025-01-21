from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from imblearn.ensemble import RUSBoostClassifier  # Import RUSBoost from imbalanced-learn library
from sklearn.model_selection import StratifiedKFold
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


def plot_roc_curve(fpr, tpr, auc_score, name="Classifier"):
    """Plot ROC curve

    :param fpr: False Positive Rate
    :param tpr: True Positive Rate
    :param auc_score: Area Under Curve (AUC)
    :param name: Name of the classifier
    """
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.4f})'.format(auc_score))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {name}')
    plt.legend(loc="lower right")
    plt.show()

def prepare_rus(X: np.array, y: np.array, minority=None, verbose: bool = False):
    """Preparing Data for Ensemble
    Make the data binary by minority class in the dataset

    :param X: np.array (n_samples, n_features)
        feature matrix

    :param y: np.array (n_samples,)
        label vector

    :param minority: int or str (default = None)
        label of minority class
        if you want to set a specific class to be minority class

    :param verbose: bool (default = False)
        verbosity

    :return: np.array, np.array
        X, y returned
    """

    # Get classes and number of them
    classes, counts = np.unique(y, return_counts=True)
    print(classes)
    print(counts)

    if minority is None:
        # Find minority class
        minority = classes[np.argmin(counts)]

    if minority not in classes:
        raise ValueError("class '{}' does not exist".format(
            minority
        ))

    # Set new label for data (1 for minority class and -1 for rest of data)
    y_ = np.where(y == minority, 1, 0)

    if verbose:
        information = "[Preparing]\n" \
                      "+ #classes: {}\n" \
                      "+ classes and counts: {}\n" \
                      "+ Minority class: {}\n" \
                      "+ Size of Minority: {}\n" \
                      "+ Size of Majority: {}\n" \
                      "".format(len(classes),
                                list(zip(classes, counts)),
                                minority,
                                np.sum(y_ == 1),
                                np.sum(y_ != 1),
                                )

        print(information)

    return X, y_

def evaluate_rus(
        name,
        base_classifier,
        X,
        y,
        minority_class=None,
        k: int = 5,
        n_runs: int = 20,
        random_state: int = None,
        verbose: bool = False,
        output_file=None, 
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
    if output_file is None:
        raise ValueError("Output file path must be specified!")

    # Open the output file in append mode
    with open(output_file, 'a') as file:
        file.write("\n")
        file.write(f"======[Dataset: {name}]======\n")

    np.random.seed(random_state)

    # Output template
    OUTPUT = "[{}] Accuracy: {:.4f}, AUC: {:.4f}"

    # Prepare the data (Make it Binary)
    X, y = prepare_rus(X, y, minority_class, verbose)

    best_roc_data = None
    best_metrics = [-np.inf, -np.inf]  # [accuracy, AUC]

    folds = np.zeros((n_runs, 2))

    start_time = time.time()

    for run in tqdm(range(n_runs)):

        # Applying k-Fold cross-validation (Stratified K-Fold)
        kFold = StratifiedKFold(n_splits=k, shuffle=True)

        metrics = np.zeros((k, 2))
        fpr_list =  []
        tpr_list = []
        # Store metrics in this variable
        for fold, (trIndexes, tsIndexes) in enumerate(kFold.split(X, y)):
            # Split data into training and test sets
            Xtr, ytr = X[trIndexes], y[trIndexes]
            Xts, yts = X[tsIndexes], y[tsIndexes]

            # print("NaN in Xtr:", np.isnan(Xtr).any())
            # print("NaN in ytr:", np.isnan(ytr).any())
            # print("Infinity in Xtr:", np.isinf(Xtr).any())
            # print("Infinity in ytr:", np.isinf(ytr).any())
            # print("Max value in Xtr:", np.max(Xtr))
            # from sklearn.preprocessing import StandardScaler
            # scaler = StandardScaler()
            # Xtr = scaler.fit_transform(Xtr)

            # Define the RUSBoost model
            model = RUSBoostClassifier(
                base_estimator=base_classifier,
                random_state=random_state,
                **kwargs
            )

            model.fit(Xtr, ytr)

            # Accuracy evaluation
            predicted = model.predict(Xts)
            auc_score = roc_auc_score(yts, predicted)
            accuracy = accuracy_score(yts, predicted)

            # Collect ROC curve data
            fpr, tpr, _ = roc_curve(yts, predicted)
            # print(fpr)
            # assert 0
            if len(fpr_list)>0:
                if len(fpr)!=len(fpr_list[0]):
                    continue
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            metrics[fold, :] = [accuracy, auc_score]
        
        run_metrics = np.mean(metrics, axis=0)
        final_fpr = np.mean(np.vstack(fpr_list), axis=0)
        final_tpr = np.mean(np.vstack(tpr_list), axis=0)
        folds[run, :] = run_metrics

        if np.all(run_metrics > best_metrics):
            best_metrics = run_metrics
            best_roc_data = (final_fpr, final_tpr, run_metrics[1])

    # End timing the loop
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert seconds to milliseconds
    tqdm.write(f"Run completed in {elapsed_time:.2f} ms")

    print()
    print(OUTPUT.format(
        "Best",
        *np.max(folds, axis=0)
    ))

    print()
    print(OUTPUT.format(
        "Best",
        *best_metrics
    ))

    with open(output_file, 'a') as file:
        file.write(OUTPUT.format("Best", *np.max(folds, axis=0)) + "\n")
        file.write(OUTPUT.format("Best", *best_metrics) + "\n")

    return best_roc_data

