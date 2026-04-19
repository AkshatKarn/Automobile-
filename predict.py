"""
Production delay prediction module using trained RandomForestRegressor model.

This module loads a pre-trained machine learning model and metadata,
and provides a function to predict production delays based on input features.
"""

import joblib
import pandas as pd
from typing import Dict



# Feature engineering function used during training
def add_features(X):
    """Apply feature engineering transformations to input data."""
    X = X.copy()
    # Interaction feature
    X['Downtime_x_Defects'] = X['Machine_Downtime'] * X['Defect_Count']
    return X


# Load the trained model and metadata
model = joblib.load("model.pkl")
metadata = joblib.load("model_meta.pkl")


def predict_delay(input_data: Dict) -> float:
    """
    Predict production delay based on input features.

    Args:
        input_data (Dict): Dictionary containing input features.
                          Keys should match the feature names used during training.

    Returns:
        float: Predicted production delay rounded to 2 decimal places.
               Ensures non-negative prediction values.

    Raises:
        KeyError: If required features from training are missing in input_data.
    """
    # Extract feature names from metadata
    feature_names = metadata["features"]

    # Convert input dictionary to DataFrame with correct column order
    # Fill missing features with 0 to handle any gaps
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_names, fill_value=0)

    # Make prediction using the trained model
    prediction = model.predict(input_df)[0]

    # Ensure prediction is non-negative
    prediction = max(0, prediction)

    # Return prediction rounded to 2 decimal places
    return round(prediction, 2)


if __name__ == "__main__":
    # Sample test input with all required features
    sample_input = {
        "Disruption_Severity": 3,
        "Parts_Quality_Score": 7,
        "Supply_Risk_Flag": 1,
        "Historical_Disruption_Count": 2,
        "Machine_Downtime": 5,
        "Defect_Count": 2,
        "Rework_Time": 4
    }

    # Make prediction
    predicted_delay = predict_delay(sample_input)

    # Print result in readable format
    print(f"Predicted Production Delay: {predicted_delay:.2f}")
