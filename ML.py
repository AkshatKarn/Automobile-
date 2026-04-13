import pandas as pd

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------------------
# 1. LOAD PROCESSED DATA
# -------------------------------
df = pd.read_csv("C:/Users/aksha/OneDrive/Desktop/HCL/processed_data.csv")

# -------------------------------
# 2. CHECK DATA (DEBUG - OPTIONAL)
# -------------------------------
print("Columns in dataset:")
print(df.columns)

# -------------------------------
# 3. FEATURES & TARGET
# -------------------------------
X = df[['Machine_Downtime', 'Defect_Count', 'Rework_Time']]
y = df['Final_Production_Delay']

# -------------------------------
# 4. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. MODEL TRAINING
# -------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------
# 6. PREDICTION
# -------------------------------
predictions = model.predict(X_test)

# -------------------------------
# 7. EVALUATION
# -------------------------------
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n📊 Model Performance:")
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

# -------------------------------
# 8. SAMPLE PREDICTION
# -------------------------------
sample = pd.DataFrame({
    'Machine_Downtime': [5],
    'Defect_Count': [2],
    'Rework_Time': [4]
})

predicted_delay = model.predict(sample)

print("\n🔮 Sample Prediction:")
print("Predicted Production Delay:", predicted_delay[0])