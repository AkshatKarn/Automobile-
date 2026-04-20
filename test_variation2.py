from feature_engineering import add_features
from predict import predict_delay
import numpy as np

# Use the working sample as starting point
working_sample = {
    "Disruption_Severity": 3,
    "Parts_Quality_Score": 7,
    "Supply_Risk_Flag": 1,
    "Historical_Disruption_Count": 2,
    "Machine_Downtime": 5,
    "Defect_Count": 2,
    "Rework_Time": 4
}

print('Working sample baseline:', predict_delay(working_sample))

print('\n=== Testing Severity (0 to 10) ===')
for val in np.arange(0, 11, 1):
    b = working_sample.copy()
    b["Disruption_Severity"] = val
    result = predict_delay(b)
    print(f'Severity {val:2.0f}: {result:.2f}')

print('\n=== Testing Quality (0 to 10) ===')
for val in np.arange(0, 11, 1):
    b = working_sample.copy()
    b["Parts_Quality_Score"] = val
    result = predict_delay(b)
    print(f'Quality {val:2.0f}: {result:.2f}')

print('\n=== Testing Downtime (0 to 30) ===')
for val in np.arange(0, 31, 3):
    b = working_sample.copy()
    b["Machine_Downtime"] = val
    result = predict_delay(b)
    print(f'Downtime {val:2.0f}: {result:.2f}')

print('\n=== Testing Defects (0 to 30) ===')
for val in np.arange(0, 31, 3):
    b = working_sample.copy()
    b["Defect_Count"] = val
    result = predict_delay(b)
    print(f'Defects {val:2.0f}: {result:.2f}')
