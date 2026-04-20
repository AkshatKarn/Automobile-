#!/usr/bin/env python
"""Test that predictions work with Supply_Risk_Flag=1"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

# This is critical - import feature_engineering into __main__ 
# so the pickle can find add_features
from feature_engineering import add_features

# Now import predict (which also imports add_features)
from predict import predict_delay
import numpy as np

# Test Advanced Analysis with Supply_Risk_Flag=1
base_input = {
    'Disruption_Severity': 3,
    'Parts_Quality_Score': 7,
    'Supply_Risk_Flag': 1,
    'Historical_Disruption_Count': 2,
    'Machine_Downtime': 5,
    'Defect_Count': 2,
    'Rework_Time': 4
}

print('✅ Testing Severity Impact (Supply_Risk_Flag=1):')
severities = np.arange(0, 11, 2)
for sev in severities:
    test_input = base_input.copy()
    test_input['Disruption_Severity'] = sev
    test_input['Historical_Disruption_Count'] = max(0, 2 + (sev/2))
    test_input['Machine_Downtime'] = 5 + (sev * 1.5)
    delay = predict_delay(test_input)
    print(f'Severity={sev:2} -> {delay:.2f} days')

print('\n✅ Testing Comparison Scenarios:')
scenarios = {
    "Best Case": {
        "Disruption_Severity": 1,
        "Parts_Quality_Score": 10,
        "Supply_Risk_Flag": 1,
        "Historical_Disruption_Count": 0,
        "Machine_Downtime": 0,
        "Defect_Count": 0,
        "Rework_Time": 0
    },
    "Average Case": {
        "Disruption_Severity": 5,
        "Parts_Quality_Score": 5,
        "Supply_Risk_Flag": 1,
        "Historical_Disruption_Count": 5,
        "Machine_Downtime": 10,
        "Defect_Count": 5,
        "Rework_Time": 5
    },
    "Worst Case": {
        "Disruption_Severity": 10,
        "Parts_Quality_Score": 0,
        "Supply_Risk_Flag": 1,
        "Historical_Disruption_Count": 20,
        "Machine_Downtime": 40,
        "Defect_Count": 30,
        "Rework_Time": 30
    }
}

for scenario_name, params in scenarios.items():
    delay = predict_delay(params)
    print(f'{scenario_name:15} -> {delay:6.2f} days')
