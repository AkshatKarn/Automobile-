from feature_engineering import add_features
from predict import predict_delay
import numpy as np

base_input = {
    "Disruption_Severity": 3,
    "Parts_Quality_Score": 7,
    "Supply_Risk_Flag": 0,
    "Historical_Disruption_Count": 2,
    "Machine_Downtime": 5,
    "Defect_Count": 2,
    "Rework_Time": 4
}

print('=== Severity with correlated factors (0-10) ===')
for sev in np.arange(0, 11, 1):
    b = base_input.copy()
    b["Disruption_Severity"] = sev
    b["Historical_Disruption_Count"] = max(0, 2 + (sev/2))
    b["Machine_Downtime"] = 5 + (sev * 1.5)
    result = predict_delay(b)
    print(f'Severity {sev:2.0f}: {result:.2f}')

print('\n=== Quality with defects inverse (0-10) ===')
for qual in np.arange(0, 11, 1):
    b = base_input.copy()
    b["Parts_Quality_Score"] = qual
    b["Defect_Count"] = max(0, 20 - (qual * 2))
    result = predict_delay(b)
    print(f'Quality {qual:2.0f}: {result:.2f}')

print('\n=== Downtime with rework (0-50) ===')
for dt in [0, 10, 20, 30, 40, 50]:
    b = base_input.copy()
    b["Machine_Downtime"] = dt
    b["Rework_Time"] = 4 + (dt * 0.3)
    result = predict_delay(b)
    print(f'Downtime {dt:2.0f}: {result:.2f}')

print('\n=== Defects with rework (0-50) ===')
for def_count in [0, 10, 20, 30, 40, 50]:
    b = base_input.copy()
    b["Defect_Count"] = def_count
    b["Rework_Time"] = 4 + (def_count * 0.4)
    result = predict_delay(b)
    print(f'Defects {def_count:2.0f}: {result:.2f}')
