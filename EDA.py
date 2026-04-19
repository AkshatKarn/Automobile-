import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------
# 1. LOAD DATA
# -------------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df


# -------------------------------
# 2. RENAME COLUMNS
# -------------------------------
def rename_columns(df):
    df = df.rename(columns={
        "Order_ID": "Vehicle_ID",
        "Supplier_ID": "Parts_Supplier_ID",
        "Product_Category": "Vehicle_Type",
        "Order_Date": "Production_Start_Date",
        "Delivery_Date": "Production_End_Date",
        "Delay_Days": "Production_Delay",
        "Disruption_Type": "Production_Issue",
        "Supplier_Reliability_Score": "Parts_Quality_Score"
    })
    return df


# -------------------------------
# 3. FEATURE ENGINEERING (NO RANDOM)
# -------------------------------
def feature_engineering(df):

    # FIX DATA TYPES
    df['Production_Delay'] = pd.to_numeric(df['Production_Delay'], errors='coerce')
    df['Disruption_Severity'] = pd.to_numeric(df['Disruption_Severity'], errors='coerce')
    df['Parts_Quality_Score'] = pd.to_numeric(df['Parts_Quality_Score'], errors='coerce')

    df.fillna(0, inplace=True)

    # FEATURES
    df['Machine_Downtime'] = (
    df['Disruption_Severity'] * df['Historical_Disruption_Count']
)
    df['Defect_Count'] = (10 - df['Parts_Quality_Score']).clip(lower=0)
    df['Rework_Time'] = df['Defect_Count'] * 1.5

    df['Final_Production_Delay'] = (
        df['Production_Delay'] +
        df['Machine_Downtime'] +
        df['Rework_Time']
    )

    return df


# -------------------------------
# 4. SAVE DATA
# -------------------------------
def save_data(df, path):
    df.to_csv(path, index=False)
    print("✅ Processed file saved successfully")


# -------------------------------
# 5. EDA VISUALS (OPTIONAL)
# -------------------------------
def run_eda(df):

    # Distribution
    plt.figure()
    plt.hist(df['Final_Production_Delay'])
    plt.title("Final Production Delay Distribution")
    plt.xlabel("Delay")
    plt.ylabel("Frequency")
    plt.show()

    # Issue vs Delay
    plt.figure()
    sns.barplot(x='Production_Issue', y='Final_Production_Delay', data=df)
    plt.xticks(rotation=45)
    plt.show()

    # Quality vs Delay
    plt.figure()
    sns.scatterplot(x='Parts_Quality_Score', y='Final_Production_Delay', data=df)
    plt.show()

    # Vehicle vs Delay
    plt.figure()
    sns.boxplot(x='Vehicle_Type', y='Final_Production_Delay', data=df)
    plt.show()


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def run_pipeline():

    raw_path = "C:/Users/aksha/OneDrive/Desktop/HCL/data.csv"
    processed_path = "C:/Users/aksha/OneDrive/Desktop/HCL/processed_data.csv"

    df = load_data(raw_path)
    df = rename_columns(df)
    df = feature_engineering(df)

    save_data(df, processed_path)

    # Optional
    run_eda(df)


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    run_pipeline()