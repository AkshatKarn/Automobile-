import pandas as pd
import joblib

# ML
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("C:/Users/aksha/OneDrive/Desktop/HCL/processed_data.csv")

# -------------------------------
# 2. BASIC CLEANING
# -------------------------------
numeric_cols = [
    'Production_Delay',
    'Disruption_Severity',
    'Parts_Quality_Score',
    'Supply_Risk_Flag',
    'Historical_Disruption_Count',
    'Machine_Downtime',
    'Defect_Count',
    'Rework_Time'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(0, inplace=True)

# -------------------------------
# 3. FEATURE ENGINEERING (inside pipeline)
# -------------------------------
def add_features(X):
    X = X.copy()
    # interaction feature (cheap boost)
    X['Downtime_x_Defects'] = X['Machine_Downtime'] * X['Defect_Count']
    return X

base_features = [
    'Disruption_Severity',
    'Parts_Quality_Score',
    'Supply_Risk_Flag',
    'Historical_Disruption_Count',
    'Machine_Downtime',
    'Defect_Count',
    'Rework_Time'
]

X = df[base_features]
y = df['Production_Delay']

# -------------------------------
# 4. TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5. PIPELINE + MODEL (TUNED)
# -------------------------------
pipeline = Pipeline(steps=[
    ("feat_eng", FunctionTransformer(add_features, validate=False)),
    ("model", RandomForestRegressor(
        n_estimators=400,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

pipeline.fit(X_train, y_train)

# -------------------------------
# 6. EVALUATION
# -------------------------------
preds = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n📊 Model Performance:")
print("MAE:", round(mae, 2))
print("R2:", round(r2, 3))

# Cross-validation (stability)
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2', n_jobs=-1)
print("CV R2 (mean):", round(cv_scores.mean(), 3))

# -------------------------------
# 7. FEATURE IMPORTANCE (after pipeline)
# -------------------------------
# get feature names after transformer
X_fe = add_features(X)
feat_names = X_fe.columns

model = pipeline.named_steps["model"]
importances = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)

print("\n🔎 Feature Importance:\n", importances.head(10))

# -------------------------------
# 8. SAVE MODEL + METADATA
# -------------------------------
joblib.dump(pipeline, "model.pkl")
joblib.dump({
    "features": base_features,
    "engineered_features": list(feat_names),
    "target": "Production_Delay"
}, "model_meta.pkl")

print("\n✅ Model & metadata saved")

# -------------------------------
# 9. SAMPLE TEST (SAFE OUTPUT)
# -------------------------------
sample = pd.DataFrame({
    'Disruption_Severity': [3],
    'Parts_Quality_Score': [7],
    'Supply_Risk_Flag': [1],
    'Historical_Disruption_Count': [2],
    'Machine_Downtime': [5],
    'Defect_Count': [2],
    'Rework_Time': [4]
})

pred = pipeline.predict(sample)[0]
pred = max(0, pred)

print("\n🔮 Sample Prediction:", round(pred, 2))