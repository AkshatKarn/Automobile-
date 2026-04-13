import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/Users/aksha/OneDrive/Desktop/HCL/data.csv")
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

# Machine downtime (random realistic)
df['Machine_Downtime'] = np.random.randint(1, 10, size=len(df))

# Defects
df['Defect_Count'] = np.random.randint(0, 5, size=len(df))

# Rework time (logic based)
df['Rework_Time'] = df['Defect_Count'] * 2


df['Final_Production_Delay'] = (
    df['Production_Delay'] + 
    df['Machine_Downtime'] + 
    df['Rework_Time']
)

# -------------------------------
# SAVE PROCESSED DATA (IMPORTANT 🔥)
# -------------------------------
df.to_csv("C:/Users/aksha/OneDrive/Desktop/HCL/processed_data.csv", index=False)

print("✅ Processed file saved successfully")


plt.hist(df['Final_Production_Delay'])
plt.title("Final Production Delay Distribution")
plt.xlabel("Delay")
plt.ylabel("Frequency")
plt.show()

sns.barplot(x='Production_Issue', y='Final_Production_Delay', data=df)
plt.xticks(rotation=45)
plt.show()

sns.scatterplot(x='Parts_Quality_Score', y='Final_Production_Delay', data=df)
plt.show()

sns.boxplot(x='Vehicle_Type', y='Final_Production_Delay', data=df)
plt.show()

