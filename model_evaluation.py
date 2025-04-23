import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

def evaluate_models(models, X_test, y_test):
    """
    Evaluate multiple models and return performance metrics.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (scipy.sparse.csr.csr_matrix): The vectorized test features
        y_test (numpy.ndarray): The test labels
    
    Returns:
        pandas.DataFrame: DataFrame containing performance metrics for each model
    """
    metrics = {}
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store metrics
        metrics[model_name] = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    return metrics

def plot_confusion_matrices(models, X_test, y_test):
    """
    Plot confusion matrices for multiple models.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (scipy.sparse.csr.csr_matrix): The vectorized test features
        y_test (numpy.ndarray): The test labels
    
    Returns:
        matplotlib.figure.Figure: Figure containing confusion matrices
    """
    n_models = len(models)
    
    # Create figure
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
    
    # Handle case with only one model
    if n_models == 1:
        axes = [axes]
    
    for ax, (model_name, model) in zip(axes, models.items()):
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{model_name}")
        
        # Add labels
        classes = ['Ham', 'Spam']
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        
        # Add values to cells
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j],
                       horizontalalignment="center",
                       verticalalignment="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    return fig

def plot_roc_curves(models, X_test, y_test):
    """
    Plot ROC curves for multiple models.
    
    Args:
        models (dict): Dictionary of trained models
        X_test (scipy.sparse.csr.csr_matrix): The vectorized test features
        y_test (numpy.ndarray): The test labels
    
    Returns:
        matplotlib.figure.Figure: Figure containing ROC curves
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors for different models
    colors = ['blue', 'red', 'green', 'purple', 'orange']
    
    for (model_name, model), color in zip(models.items(), colors):
        try:
            # Get prediction probabilities
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            # Plot ROC curve
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.2f})')
        except Exception as e:
            print(f"Could not plot ROC curve for {model_name}: {str(e)}")
    
    # Add random guess line
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    # Set labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    return fig
