from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from ensemble import HashBasedUndersamplingEnsemble
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


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

def prepare(X: np.array, y: np.array, minority=None, verbose: bool = False):
    """Preparing Data for Ensemble
    Make the data binary by minority class in the dataset

    :param X: np.array (n_samples, n_features)
        feature matrix

    :param y: np.array (n_samples,)
        label vecotr

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

    if minority is None:
        # find minority class
        minority = classes[np.argmin(counts)]

    if minority not in classes:
        raise ValueError("class '{}' does not exist".format(
            minority
        ))

    # set new label for data (1 for minority class and -1 for rest of data)
    y_ = np.where(y == minority, 1, -1)

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


def evaluate(
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
    """Model Evaluation with ROC curve plotting capabilities

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

    :return List of ROC data (fprs, tprs, aucs)
    """

    print()
    print("======[Dataset: {}]======".format(
        name
    ))

    np.random.seed(random_state)

    # Output template
    OUTPUT = "[{}] Accuracy: {:.4f}, AUC: {:.4f}"

    # Prepare the data (Make it Binary)
    X, y = prepare(X, y, minority_class, verbose)

    best_roc_data = None
    best_metrics = [-np.inf, -np.inf]  # [accuracy, AUC]

    folds = np.zeros((n_runs, 2))

    # k-Fold (k = 5 as per the paper)
    for run in tqdm(range(n_runs)):

        # Applying k-Fold (k = 5 due to the paper)
        kFold = StratifiedKFold(n_splits=k, shuffle=True)

        # store metrics in this variable
        metrics = np.zeros((k, 2))
        fpr_list =  []
        tpr_list = []
        for fold, (trIndexes, tsIndexes) in enumerate(kFold.split(X, y)):
            # Split data to Train and Test
            Xtr, ytr = X[trIndexes], y[trIndexes]
            Xts, yts = X[tsIndexes], y[tsIndexes]

            # Define Model
            model = HashBasedUndersamplingEnsemble(
                base_estimator=base_classifier,
                n_iterations=n_iterations,
                random_state=random_state,
                **kwargs
            )

            # Fit the training data on the model
            model.fit(Xtr, ytr)

            # Predict the test data and get predicted probabilities
            # y_prob = model.predict_proba(Xts)[:, 1]  # Probabilities for positive class
            predicted = model.predict(Xts)

            # AUC evaluation
            # auc_score = roc_auc_score(yts, y_prob)
            AUC = roc_auc_score(yts, predicted)
            # Accuracy evaluation
            accuracy = accuracy_score(yts, predicted)

            fpr, tpr, _ = roc_curve(yts, predicted)
            fpr_list.append(fpr)
            tpr_list.append(tpr)

            # # Collect ROC curve data
            # fpr, tpr, _ = roc_curve(yts, y_prob)
            # roc_data.append((fpr, tpr, auc_score))
            metrics[fold, :] = [accuracy, AUC]

        # folds[run, :] = np.mean(metrics, axis=0)
        run_metrics = np.mean(metrics, axis=0)
        final_fpr = np.mean(np.vstack(fpr_list), axis=0)
        final_tpr = np.mean(np.vstack(tpr_list), axis=0)
        folds[run, :] = run_metrics

        # Check if this is the best run
        if np.all(run_metrics > best_metrics):
            best_metrics = run_metrics
            best_roc_data = (final_fpr, final_tpr, run_metrics[1])

    # print(OUTPUT.format("Best", accuracy, auc_score))
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

    # if best_roc_data:
    #     fpr, tpr, auc_score = best_roc_data
    #     plot_roc_curve(fpr, tpr, auc_score, name=base_classifier)

    # Return ROC data for the best run
    return best_roc_data