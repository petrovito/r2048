from typing import Dict, Any

import torch
import numpy as np


def compute_metrics(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    """Compute various metrics for model evaluation.
    
    Args:
        outputs: Model outputs of shape (batch_size, num_classes)
        targets: Target values of shape (batch_size, num_classes)
        
    Returns:
        Dictionary of metric names and values
    """
    # Convert to numpy for easier computation
    outputs_np = outputs.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    
    # Get predictions
    preds = np.argmax(outputs_np, axis=1)
    true_labels = np.argmax(targets_np, axis=1)
    
    # Compute accuracy
    accuracy = np.mean(preds == true_labels)
    
    # Compute per-class accuracy
    class_accuracies = {}
    for i in range(outputs_np.shape[1]):
        mask = true_labels == i
        if np.any(mask):
            class_accuracies[f"class_{i}_acc"] = np.mean(preds[mask] == true_labels[mask])
    
    # Compute confusion matrix
    confusion_matrix = np.zeros((outputs_np.shape[1], outputs_np.shape[1]))
    for t, p in zip(true_labels, preds):
        confusion_matrix[t, p] += 1
    
    # Compute precision and recall
    precision = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)
    recall = np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)
    
    # Compute F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    
    metrics = {
        "accuracy": float(accuracy),
        "mean_precision": float(np.mean(precision)),
        "mean_recall": float(np.mean(recall)),
        "mean_f1": float(np.mean(f1)),
        **{f"class_{i}_acc": float(acc) for i, acc in class_accuracies.items()},
    }
    
    return metrics 