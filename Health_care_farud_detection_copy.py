import streamlit as st
import pandas as pd
import joblib
from preprocess import preprocess_input

# Load the model
model = joblib.load('Healthcare_fraud_detection_copy.pkl')

st.title("Health Care Fraud Detection")

# Select input mode
input_mode = st.radio("Select input method", ("Upload CSV file", "Manual input"))

if input_mode == "Upload CSV file":
    uploaded_file = st.file_uploader("Upload healthcare claim data (CSV)")
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Raw input data", data)

        # Preprocess
        processed_data = preprocess_input(data)

        if st.button("Predict Fraud"):
            predictions = model.predict(processed_data)
            prediction_labels = ["Fraud" if pred == 1 else "Not Fraud" for pred in predictions]
            
            # Create a DataFrame with only the predictions
            result_df = pd.DataFrame({'Fraud Prediction': prediction_labels})
            st.dataframe(result_df)

else:
    st.subheader("Enter claim details manually")

    # Form fields
    provider = st.text_input("Enter ProviderID (Eg: PRV51002)")
    beneficiar = st.text_input("Enter BeneficiaryID (Eg: BENE14252)")
    ClaimStartDt = st.text_input("Enter ClaimStartDt")
    ClaimEndDt = st.text_input("Enter ClaimEndDt")
    amt_reimbursed = st.number_input("Enter claim amount reimbursed")
    phyID = st.text_input("Enter Attending PhysicianID")
    AdmissionDt = st.text_input("Enter AdmissionDt")
    DeductibleAmtPaid = st.number_input("Enter DeductibleAmtPaid")
    DischargeDt = st.text_input("Enter DischargeDt")
    ClmDiagnosisCode_1 = st.number_input("Enter ClmDiagnosisCode_1")
    DOB = st.text_input("Enter DOB")
    Gender = st.selectbox("Select Gender (1-Male, 2-Female)", [1, 2])
    Race = st.selectbox("Select Race", [1, 2, 3, 4])
    RenalDiseaseIndicator = st.selectbox("Select RenalDiseaseIndicator", [0, 1])
    State = st.selectbox("Select State", [1, 2, 3, 4, 5, 6, 7])
    county = st.number_input("Enter County Number")

    st.subheader("Diseases")
    chronic_alz = st.checkbox("Alzheimer's")
    chronic_heart = st.checkbox("Heart Failure")
    chronic_kidney = st.checkbox("Kidney Disease")
    chronic_cancer = st.checkbox("Cancer")
    chronic_pulmonary = st.checkbox("Obstructive Pulmonary Disease")
    chronic_depression = st.checkbox("Depression")
    chronic_diabetes = st.checkbox("Diabetes")
    chronic_ischemic = st.checkbox("Ischemic Heart Disease")
    chronic_osteoporosis = st.checkbox("Osteoporosis")
    chronic_arthritis = st.checkbox("Rheumatoid Arthritis")
    chronic_stroke = st.checkbox("Stroke")
    
    manual_input = pd.DataFrame({
        'Provider': [provider],
        'BeneID': [beneficiar],
        'InscClaimAmtReimbursed': [amt_reimbursed],
        'AttendingPhysician': [phyID],
        'DeductibleAmtPaid': [DeductibleAmtPaid],
        'ClmDiagnosisCode_1': [ClmDiagnosisCode_1],
        'Gender': [Gender],
        'Race': [Race],
        'RenalDiseaseIndicator': [RenalDiseaseIndicator],
        'State': [State],
        'County': [county],
        'ClaimStartDt': [ClaimStartDt],
        'ClaimEndDt': [ClaimEndDt],
        'AdmissionDt': [AdmissionDt],
        'DischargeDt': [DischargeDt],
        'DOB': [DOB],
        'ChronicCond_Alzheimer': [1 if chronic_alz else 0],
        'ChronicCond_Heartfailure': [1 if chronic_heart else 0],
        'ChronicCond_KidneyDisease': [1 if chronic_kidney else 0],
        'ChronicCond_Cancer': [1 if chronic_cancer else 0],
        'ChronicCond_ObstrPulmonary': [1 if chronic_pulmonary else 0],
        'ChronicCond_Depression': [1 if chronic_depression else 0],
        'ChronicCond_Diabetes': [1 if chronic_diabetes else 0],
        'ChronicCond_IschemicHeart': [1 if chronic_ischemic else 0],
        'ChronicCond_Osteoporasis': [1 if chronic_osteoporosis else 0],
        'ChronicCond_rheumatoidarthritis': [1 if chronic_arthritis else 0],
        'ChronicCond_stroke': [1 if chronic_stroke else 0],
    })

    st.write("Input Summary:", manual_input)

    if st.button("Predict Fraud (Manual)"):
        processed_manual = preprocess_input(manual_input)
        predictions = model.predict(processed_manual)
        
        prediction_text = "Fraud" if predictions[0] == 1 else "Not Fraud"
        
        st.write("Fraud Prediction:")
        st.success(prediction_text)  # Green success message
