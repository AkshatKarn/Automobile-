from feature_engineering import add_features
import joblib
import pandas as pd
import numpy as np

model = joblib.load('model.pkl')
metadata = joblib.load('model_meta.pkl')
features = metadata['features']

base_input = {
    "Disruption_Severity": 3,
    "Parts_Quality_Score": 7,
    "Supply_Risk_Flag": 0,
    "Historical_Disruption_Count": 2,
    "Machine_Downtime": 5,
    "Defect_Count": 2,
    "Rework_Time": 4
}

print('=== Raw predictions (without max(0)) ===')
print('\nSeverity with correlated factors:')
for sev in [0, 3, 6, 10]:
    b = base_input.copy()
    b["Disruption_Severity"] = sev
    b["Historical_Disruption_Count"] = max(0, 2 + (sev/2))
    b["Machine_Downtime"] = 5 + (sev * 1.5)
    df = pd.DataFrame([b]).reindex(columns=features, fill_value=0)
    raw_pred = model.predict(df)[0]
    print(f'  Severity {sev}: raw={raw_pred:.2f}, clamped={max(0, raw_pred):.2f}')

print('\nQuality with defects:')
for qual in [0, 5, 10]:
    b = base_input.copy()
    b["Parts_Quality_Score"] = qual
    b["Defect_Count"] = max(0, 20 - (qual * 2))
    df = pd.DataFrame([b]).reindex(columns=features, fill_value=0)
    raw_pred = model.predict(df)[0]
    print(f'  Quality {qual}: raw={raw_pred:.2f}, clamped={max(0, raw_pred):.2f}')

print('\nDowntime with rework:')
for dt in [0, 25, 50]:
    b = base_input.copy()
    b["Machine_Downtime"] = dt
    b["Rework_Time"] = 4 + (dt * 0.3)
    df = pd.DataFrame([b]).reindex(columns=features, fill_value=0)
    raw_pred = model.predict(df)[0]
    print(f'  Downtime {dt}: raw={raw_pred:.2f}, clamped={max(0, raw_pred):.2f}')
