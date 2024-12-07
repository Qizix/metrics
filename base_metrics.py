import numpy as np
import pandas as pd

def rmse(y_true, y_pred):
    """
    Calculate the Root Mean Square Error (RMSE).
    
    Parameters:
    y_true (list or np.array): Actual values.
    y_pred (list or np.array): Predicted values.
    
    Returns:
    float: RMSE value.
    """