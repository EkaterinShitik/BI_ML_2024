import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    total_number = y_pred.shape[0]
    true_all = (y_pred == y_true).sum()
    true_pos = float(((y_pred == 1) & (y_true == 1)).sum())
    false_pos = float(((y_pred == 1) & (y_true == 0)).sum())
    false_neg = float(((y_pred == 0) & (y_true == 1)).sum())
    accuracy = true_all / total_number
    precision_denom = (true_pos + false_pos)
    precision = np.divide(true_pos, precision_denom, out=np.zeros_like(true_pos), where=precision_denom!=0)
    recall_denom = (true_pos + false_neg)
    recall = np.divide(true_pos, recall_denom, out=np.zeros_like(true_pos), where=recall_denom!=0)
    f1_score_numer = 2 * (precision * recall)
    f1_score_denom = (precision + recall)
    f1_score = np.divide(f1_score_numer, f1_score_denom, out=np.zeros_like(f1_score_numer), where=f1_score_denom!=0)
    print(f"Accuracy is {accuracy}")
    print(f"Precision is {precision}")
    print(f"Recall is {recall}")
    print(f"F1_score is {f1_score}")


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """
    pass


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    """
    YOUR CODE IS HERE
    """
    pass


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    """
    YOUR CODE IS HERE
    """
    pass


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    """
    YOUR CODE IS HERE
    """
    pass
    