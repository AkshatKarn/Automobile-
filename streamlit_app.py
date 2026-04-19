import streamlit as st
from predict import predict_delay

# -------------------------------
# PAGE TITLE
# -------------------------------
st.set_page_config(page_title="Production Delay Predictor")

st.title("🚗 Automobile Production Delay Predictor")

st.write("Enter production details to predict delay:")

# -------------------------------
# INPUT FIELDS
# -------------------------------
disruption = st.slider("Disruption Severity", 0, 10, 3)
quality = st.slider("Parts Quality Score", 0, 10, 7)
risk = st.selectbox("Supply Risk Flag", [0, 1])
history = st.number_input("Historical Disruption Count", 0, 10, 2)
downtime = st.number_input("Machine Downtime", 0, 20, 5)
defects = st.number_input("Defect Count", 0, 10, 2)
rework = st.number_input("Rework Time", 0, 20, 4)

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("Predict Delay"):

    input_data = {
        "Disruption_Severity": disruption,
        "Parts_Quality_Score": quality,
        "Supply_Risk_Flag": risk,
        "Historical_Disruption_Count": history,
        "Machine_Downtime": downtime,
        "Defect_Count": defects,
        "Rework_Time": rework
    }

    result = predict_delay(input_data)

    st.success(f"🚀 Predicted Production Delay: {result} days")