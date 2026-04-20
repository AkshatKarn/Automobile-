"""
Feature engineering utilities for production delay prediction model.
"""


def add_features(X):
    """Apply feature engineering transformations to input data."""
    X = X.copy()
    # Interaction feature
    X['Downtime_x_Defects'] = X['Machine_Downtime'] * X['Defect_Count']
    return X
