from feature_engineering import add_features
import joblib
import pandas as pd

model = joblib.load('model.pkl')
metadata = joblib.load('model_meta.pkl')
features = metadata['features']
print('Base features:', features)
print('Engineered features:', metadata.get('engineered_features', 'N/A'))

base_input = {
    "Disruption_Severity": 3,
    "Parts_Quality_Score": 7,
    "Supply_Risk_Flag": 0,
    "Historical_Disruption_Count": 2,
    "Machine_Downtime": 5,
    "Defect_Count": 2,
    "Rework_Time": 4
}

# Test with direct pipeline
df = pd.DataFrame([base_input]).reindex(columns=features, fill_value=0)
print('\nBefore feature engineering:')
print(df)

df_engineered = add_features(df)
print('\nAfter feature engineering:')
print(df_engineered)
print('Engineered feature (Downtime_x_Defects):', df_engineered['Downtime_x_Defects'].values[0])

# Try pipeline prediction
result = model.predict(df)[0]
print('\nPipeline prediction:', result)
