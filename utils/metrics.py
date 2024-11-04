import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, precision_score, recall_score

def multimodal_alignment_score(text_features, image_features, audio_features, alpha=0.33, beta=0.33, gamma=0.34):
    """
    Computes the multimodal alignment score between text, image, and audio features.

    :param text_features: np.array of shape (n_samples, n_features), feature vectors from the text modality
    :param image_features: np.array of shape (n_samples, n_features), feature vectors from the image modality
    :param audio_features: np.array of shape (n_samples, n_features), feature vectors from the audio modality
    :param alpha: float, weight for text-image alignment
    :param beta: float, weight for text-audio alignment
    :param gamma: float, weight for image-audio alignment
    :return: alignment score, float
    """
    text_image_alignment = np.dot(text_features, image_features.T).mean()
    text_audio_alignment = np.dot(text_features, audio_features.T).mean()
    image_audio_alignment = np.dot(image_features, audio_features.T).mean()

    alignment_score = alpha * text_image_alignment + beta * text_audio_alignment + gamma * image_audio_alignment
    return alignment_score

def classification_metrics(y_true, y_pred):
    """
    Computes various metrics for classification tasks.

    :param y_true: np.array, true labels
    :param y_pred: np.array, predicted labels
    :return: dict with accuracy, F1 score, precision, and recall
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    return {
        "accuracy": accuracy, 
        "f1_score": f1,
        "precision": precision,
        "recall": recall
    }

def regression_metrics(y_true, y_pred):
    """
    Computes various metrics for regression tasks.

    :param y_true: np.array, true values
    :param y_pred: np.array, predicted values
    :return: dict with MSE and RMSE
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    return {
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse
    }

def compute_weighted_dot_product(features_1, features_2, weight=1.0):
    """
    Computes the weighted dot product between two sets of feature vectors.

    :param features_1: np.array, first set of feature vectors
    :param features_2: np.array, second set of feature vectors
    :param weight: float, weight applied to the dot product
    :return: weighted dot product
    """
    dot_product = np.dot(features_1, features_2.T).mean()
    return weight * dot_product

def compute_alignment_score(text_features, image_features, audio_features, weights=(0.33, 0.33, 0.34)):
    """
    Wrapper function to compute multimodal alignment score with optional weighting.

    :param text_features: np.array of shape (n_samples, n_features), feature vectors from text
    :param image_features: np.array of shape (n_samples, n_features), feature vectors from images
    :param audio_features: np.array of shape (n_samples, n_features), feature vectors from audio
    :param weights: tuple of three floats, weights for text-image, text-audio, and image-audio alignments
    :return: alignment score, float
    """
    text_image_score = compute_weighted_dot_product(text_features, image_features, weight=weights[0])
    text_audio_score = compute_weighted_dot_product(text_features, audio_features, weight=weights[1])
    image_audio_score = compute_weighted_dot_product(image_features, audio_features, weight=weights[2])
    
    return text_image_score + text_audio_score + image_audio_score

def get_confusion_matrix_metrics(confusion_matrix):
    """
    Extracts metrics from a confusion matrix for classification problems.

    :param confusion_matrix: np.array, confusion matrix
    :return: dict with TP, FP, TN, FN counts
    """
    TP = np.diag(confusion_matrix)
    FP = confusion_matrix.sum(axis=0) - TP
    FN = confusion_matrix.sum(axis=1) - TP
    TN = confusion_matrix.sum() - (FP + FN + TP)

    return {
        "true_positive": TP.sum(),
        "false_positive": FP.sum(),
        "false_negative": FN.sum(),
        "true_negative": TN.sum()
    }

def evaluate_multimodal_classification(y_true_classification, y_pred_classification):
    """
    Evaluates classification metrics.

    :param y_true_classification: np.array, true labels
    :param y_pred_classification: np.array, predicted labels
    :return: dict with various classification metrics
    """
    metrics = classification_metrics(y_true_classification, y_pred_classification)
    return metrics

def evaluate_multimodal_regression(y_true_regression, y_pred_regression):
    """
    Evaluates regression metrics.

    :param y_true_regression: np.array, true values for regression task
    :param y_pred_regression: np.array, predicted values for regression task
    :return: dict with regression metrics
    """
    metrics = regression_metrics(y_true_regression, y_pred_regression)
    return metrics

def evaluate_multimodal_alignment(text_features, image_features, audio_features, weights=(0.33, 0.33, 0.34)):
    """
    Evaluates the multimodal alignment score.

    :param text_features: np.array, feature vectors from text
    :param image_features: np.array, feature vectors from images
    :param audio_features: np.array, feature vectors from audio
    :param weights: tuple, optional weights for different alignments
    :return: alignment score, float
    """
    return compute_alignment_score(text_features, image_features, audio_features, weights=weights)

def evaluate_multimodal_model(y_true_classification, y_pred_classification, y_true_regression, y_pred_regression, text_features, image_features, audio_features):
    """
    Evaluates a multimodal model with classification, regression, and alignment metrics.

    :param y_true_classification: np.array, true labels for classification task
    :param y_pred_classification: np.array, predicted labels for classification task
    :param y_true_regression: np.array, true values for regression task
    :param y_pred_regression: np.array, predicted values for regression task
    :param text_features: np.array, feature vectors from text modality
    :param image_features: np.array, feature vectors from image modality
    :param audio_features: np.array, feature vectors from audio modality
    :return: dict with all evaluation metrics
    """
    results = {}

    # Evaluate classification metrics
    classification_results = evaluate_multimodal_classification(y_true_classification, y_pred_classification)
    results.update(classification_results)

    # Evaluate regression metrics
    regression_results = evaluate_multimodal_regression(y_true_regression, y_pred_regression)
    results.update(regression_results)

    # Evaluate multimodal alignment score
    alignment_score = evaluate_multimodal_alignment(text_features, image_features, audio_features)
    results["multimodal_alignment_score"] = alignment_score

    return results

def print_metrics(metrics_dict):
    """
    Utility function to print evaluation metrics in a clean format.

    :param metrics_dict: dict, containing evaluation metrics
    """
    for metric, value in metrics_dict.items():
        print(f"{metric}: {value:.4f}")

def save_metrics(metrics_dict, file_path):
    """
    Saves the evaluation metrics to a file.

    :param metrics_dict: dict, containing evaluation metrics
    :param file_path: str, path to save the metrics file
    """
    with open(file_path, 'w') as f:
        for metric, value in metrics_dict.items():
            f.write(f"{metric}: {value:.4f}\n")

def load_metrics(file_path):
    """
    Loads evaluation metrics from a file.

    :param file_path: str, path to the file containing metrics
    :return: dict of loaded metrics
    """
    metrics = {}
    with open(file_path, 'r') as f:
        for line in f:
            metric, value = line.strip().split(": ")
            metrics[metric] = float(value)
    return metrics