from feature_engineering import add_features
from predict import predict_delay
import numpy as np

# New base input
base_input = {
    "Disruption_Severity": 10,
    "Parts_Quality_Score": 10,
    "Supply_Risk_Flag": 0,
    "Historical_Disruption_Count": 10,
    "Machine_Downtime": 30,
    "Defect_Count": 20,
    "Rework_Time": 20
}

print('=== Testing Severity (varying 0 to 10) ===')
for val in np.arange(0, 11, 1):
    b = base_input.copy()
    b["Disruption_Severity"] = val
    result = predict_delay(b)
    print(f'Severity {val:2.0f}: {result:.2f}')

print('\n=== Testing Quality (varying 0 to 10) ===')
for val in np.arange(0, 11, 1):
    b = base_input.copy()
    b["Parts_Quality_Score"] = val
    result = predict_delay(b)
    print(f'Quality {val:2.0f}: {result:.2f}')

print('\n=== Testing Downtime (varying 0 to 50) ===')
for val in np.arange(0, 51, 10):
    b = base_input.copy()
    b["Machine_Downtime"] = val
    result = predict_delay(b)
    print(f'Downtime {val:2.0f}: {result:.2f}')

print('\n=== Testing Defects (varying 0 to 50) ===')
for val in np.arange(0, 51, 10):
    b = base_input.copy()
    b["Defect_Count"] = val
    result = predict_delay(b)
    print(f'Defects {val:2.0f}: {result:.2f}')
