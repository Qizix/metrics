import numpy as np
import pandas as pd

# Regression

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE).
    
    Parameters:
    y_true (list or np.array): Actual values.
    y_pred (list or np.array): Predicted values.
    
    Returns:
    float: RMSE value.
    """
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
def mae(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE).
    Parameters:
    y_true (list or np.array): Actual values.
    y_pred (list or np.array): Predicted values.
    
    Returns:
    float: MAE value.
    """
    
    return np.mean(np.abs(y_true - y_pred)) 

 # Classification
 
def accuracy(y_true, y_pred):
    """
    Calculate the accuracy of the model.
    Parameters:
    y_true (list or np.array): Actual values.
    y_pred (list or np.array): Predicted values.
    
    Returns:
    float: Accuracy value.
    """
    
    return np.mean(y_true == y_pred)

def recall(y_true, y_pred):
    """
    Calculate the recall of the model.
    
    Parameters:
    y_true (list or np.array): Actual values (ground truth).
    y_pred (list or np.array): Predicted values.
    
    Returns:
    float: Recall value.
    """
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    if true_positives + false_negatives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_negatives)

def precicion(y_true, y_pred):
    """
    Calculate the precision of the model.
    
    Parameters:
    y_true (list or np.array): Actual values (ground truth).
    y_pred (list or np.array): Predicted values.
    
    Returns:
    float: Precision value.
    """
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    
    if true_positives + false_positive == 0:
        return 0.0
    
    return true_positives / (true_positives + false_positive)    
    
def f1(y_true, y_pred):
    """
    Calculate the f1 of the model.
    
    Parameters:
    y_true (list or np.array): Actual values (ground truth).
    y_pred (list or np.array): Predicted values.
    
    Returns:
    float: F1 value.
    """
    
    true_positives = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    
    return true_positives / (true_positives + (false_positive + false_negatives) / 2)   