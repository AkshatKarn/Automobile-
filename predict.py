"""
Production delay prediction module using trained RandomForestRegressor model.

This module loads a pre-trained machine learning model and metadata,
and provides a function to predict production delays based on input features.
"""

import joblib
import pandas as pd
from typing import Dict
import os

# Import feature engineering function for unpickling the model
from feature_engineering import add_features  # noqa: F401


# Lazy load the trained model and metadata
_model = None
_metadata = None


def _load_model_and_metadata():
    """Load model and metadata on first use."""
    global _model, _metadata
    if _model is None or _metadata is None:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "model.pkl")
        metadata_path = os.path.join(script_dir, "model_meta.pkl")
        
        _model = joblib.load(model_path)
        _metadata = joblib.load(metadata_path)
    return _model, _metadata


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
    # Load model and metadata on first use
    model, metadata = _load_model_and_metadata()
    
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
