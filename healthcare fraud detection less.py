# -*- coding: utf-8 -*-
# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# collecting training data 
train=pd.read_csv('healthcare fraud detection/Train_Outpatientdata.csv')
inpatient=pd.read_csv('healthcare fraud detection/Train_Inpatientdata.csv')
outpatient=pd.read_csv('healthcare fraud detection/Train_Outpatientdata.csv')
beneficiary=pd.read_csv('healthcare fraud detection/Train_Beneficiarydata.csv')
train

# %%
train_data_lst=[train,inpatient,outpatient,beneficiary]
data_names=["Train Data",'Inpatient Data','Outpatient Data','Beneficiary Data']
for i,j in zip(train_data_lst,data_names):
    print('\n******************',j,'**********************\n')
    print("Shape = ",i.shape)
    print("\nColumns ",i.columns)
    print('\nData Types :\n',i.dtypes)
    print("\nComplete Info : \n",i.info())
    print("\nComplete Description : \n",i.describe)

# %%
print(inpatient.columns)
print(outpatient.columns)

# %%
# Column Description

# Identifiers & General Information

# - Provider: Unique ID representing the healthcare provider.
# - PotentialFraud: Label indicating whether the provider is suspected of fraud (Yes/No).
# - BeneID: Unique ID assigned to the beneficiary (patient).
# - ClaimID: Unique identifier for an insurance claim.

# Claim Dates

# - ClaimStartDt: The date when the claim period starts.
# - ClaimEndDt: The date when the claim period ends.
# - AdmissionDt: Date when the patient was admitted (for inpatient claims).
# - DischargeDt: Date when the patient was discharged (for inpatient claims).

# Financial Information

# - InscClaimAmtReimbursed: Amount reimbursed by insurance for the claim.
# - DeductibleAmtPaid: The deductible amount paid by the patient.

# Medical Personnel Involved

# - AttendingPhysician: ID of the primary physician overseeing the treatment.
# - OperatingPhysician: ID of the physician performing an operation (if any).
# - OtherPhysician: ID of any other involved physician.

# Diagnosis & Procedures

# - ClmAdmitDiagnosisCode: Code representing the primary diagnosis at admission.
# - DiagnosisGroupCode: Group code classifying related diagnoses.
# - ClmDiagnosisCode_1 to ClmDiagnosisCode_10: Diagnosis codes related to the claim (up to 10).
# - ClmProcedureCode_1 to ClmProcedureCode_6: Procedure codes indicating medical procedures performed during treatment.

### **Patient Demographics & Health Status**

#  - DOB: Date of Birth of the patient.
#  - DOD: Date of Death (if applicable).
#  - Gender: Gender of the patient.
#  - Race: Patient’s race category.
#  - RenalDiseaseIndicator: Indicates if the patient has End-Stage Renal Disease (Y/N).
#  - State: State of residence for the patient.
#  - County: County of residence for the patient.

# Insurance Coverage Details

# - NoOfMonths_PartACov: Number of months the patient had Medicare Part A coverage.
# - NoOfMonths_PartBCov: Number of months the patient had Medicare Part B coverage.

# Chronic Conditions

# (Indicate whether the patient has a chronic condition: 1 = Yes, 0 = No)

#  - ChronicCond_Alzheimer: Alzheimer's disease.
#  - ChronicCond_Heartfailure: Heart failure.
#  - ChronicCond_KidneyDisease: Chronic kidney disease.
#  - ChronicCond_Cancer: Cancer.
#  - ChronicCond_ObstrPulmonary: Chronic obstructive pulmonary disease (COPD).
#  - ChronicCond_Depression: Depression.
#  - ChronicCond_Diabetes: Diabetes.
#  - ChronicCond_IschemicHeart: Ischemic heart disease.
#  - ChronicCond_Osteoporasis: Osteoporosis.
#  - ChronicCond_rheumatoidarthritis: Rheumatoid arthritis.
#  - ChronicCond_stroke: Stroke.

# Annual Financial Data

# - IPAnnualReimbursementAmt: Total inpatient claims reimbursement amount in the year.
# - IPAnnualDeductibleAmt: Total inpatient deductible amount paid by the patient in the year.
# - OPAnnualReimbursementAmt: Total outpatient claims reimbursement amount in the year.
# - OPAnnualDeductibleAmt: Total outpatient deductible amount paid by the patient in the year.

# %%
import warnings
warnings.filterwarnings('ignore')

# %%
# Checking missing columns 
print('Columns not in OP:',[i for i in inpatient.columns if i not in outpatient.columns])
print('Columns not in IP:',[i for i in outpatient.columns if i not in inpatient.columns])

# %%
# Merging the training datsets 
# Merging outpatient and inpatient data 
patient_merge=pd.concat([inpatient,outpatient],ignore_index=True)
patient_merge


# %%
#Merging  patient and beneficiary data
beneficiary_merge=patient_merge.merge(beneficiary,how='left',on='BeneID')
beneficiary_merge

# %%
# Merging target data to beneficiary data
train_merge=train.merge(beneficiary_merge,how='left',on='Provider')
train_merge

# %%
#Selecting test data

test=pd.read_csv('healthcare fraud detection/Test.csv')
inpatient_test=pd.read_csv('healthcare fraud detection/Test_Inpatientdata.csv')
outpatient_test=pd.read_csv('healthcare fraud detection/Test_Outpatientdata.csv')
beneficiary_test=pd.read_csv('healthcare fraud detection/Test_Beneficiarydata.csv')
test

# %%
test_data_list=[test,inpatient_test,outpatient_test,beneficiary_test]
for i,j in zip(test_data_list,data_names):
    print('\n******************',j,'**********************\n')
    print("Shape = ",i.shape)
    print("\nColumns ",i.columns)
    print('\nData Types :\n',i.dtypes)
    print("\nComplete Info : \n",i.info())
    print("\nComplete Description : \n",i.describe)

# %%

print([i for i in inpatient_test.columns if i not in outpatient_test.columns])
print([i for i in outpatient_test.columns if i not in inpatient_test.columns])

# %%
#Merging test data of inpatient and outpatient (Same columns)
patient_merge_test=pd.concat([inpatient_test,outpatient_test],ignore_index=True)
patient_merge_test

# %%
#Merging to Beneficiary and patient data
beneficiary_merge_test=patient_merge_test.merge(beneficiary_test,on='BeneID',how='left')
beneficiary_merge

# %%
#merging to test data
test_merge=test.merge(beneficiary_merge_test,how='left',on='Provider')
test_merge

# %%
# Finding the missing values in both test and train dataset 
print(train_merge.isna().sum())
print(".................................")
print(test_merge.isna().sum())

# %%
#Combining the test_merge data and Train_merge data ( THE ACTUAL DATASET )
final=pd.concat([train_merge,test_merge],ignore_index=True)
final

# %%
final.columns

# %%
final.head()

# %%
final.tail()

# %%
                 #Gathering Basic information
final.shape

# %%
final.describe()

# %%
final.info()

# %%
# Column description 
# object - 26
# numeric - 29

# %%
# Finding the correlation between numerical values 
correlation=final.corr(numeric_only=True)
plt.figure(figsize=(12,8))
sns.heatmap(correlation,fmt='.2f',cmap='viridis',linewidths=0.5,linecolor='black')
plt.show()

# %%
obj_cols=final.select_dtypes(include=object).columns
obj_cols

# %%
final.isna().sum()

# %%
# Column Dropping
# Dropping columns and reasons 
# ClmDiagnosisCode_2 to 10 - Too many missing values 
# ClmProcedureCode_1 to 6 - Too many missing values
# DiagnosisGroupCode, ClmAdmitDiagnosisCode, OtherPhysician,
# +OperatingPhysician,DOD -- Too many missing values


drop_cols=['ClmDiagnosisCode_2','ClmDiagnosisCode_3','ClmDiagnosisCode_4','ClmDiagnosisCode_5','ClmDiagnosisCode_6',
           'ClmDiagnosisCode_7','ClmDiagnosisCode_8','ClmDiagnosisCode_9','ClmDiagnosisCode_10','ClmProcedureCode_1',
           'ClmProcedureCode_2','ClmProcedureCode_3','ClmProcedureCode_4','ClmProcedureCode_5','ClmProcedureCode_6',
           'ClmAdmitDiagnosisCode','OtherPhysician','OperatingPhysician',
           'DOD']

final.drop(drop_cols,axis=1,inplace=True)

# %%
final.isna().sum()

# %%
# Finding the unique values and value counts of other columns with missing values 
print(final['AttendingPhysician'].unique())
print(final['DeductibleAmtPaid'].unique())
print(final['ClmDiagnosisCode_1'].unique())

# %%
#Filling missing values
final['AttendingPhysician'].fillna(final['AttendingPhysician'].mode()[0],inplace=True)
final['DeductibleAmtPaid'].fillna(final['DeductibleAmtPaid'].mean(),inplace=True)
final['ClmDiagnosisCode_1'].fillna(final['ClmDiagnosisCode_1'].mode()[0],inplace=True)

# %%
# task 1 - beneficiary id
# finding pattern between beneficiary id and claims 
# final['noofclaimsperBENID']=final.groupby('BeneID') ['ClaimID'].transform('count')

# %%
#Data Convertion
# columns with value 1,2
convert_cols=['ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke']
for i in convert_cols:
    final[i]=final[i].map({1:0,2:1})
#Converting RenalDiseaseIndicator to 0 and 1    
final['RenalDiseaseIndicator']=final['RenalDiseaseIndicator'].map({'Y':1,'0':0}) 

# %%
# Visualization

# %%
sns.countplot(x='PotentialFraud',data=train_merge)

# %%
sns.countplot(x='PotentialFraud',data=final)
plt.show()

# %%
# count_plot of categorical values 
categorical_cols=['Gender','Race','RenalDiseaseIndicator','ChronicCond_Alzheimer',
       'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
       'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary',
       'ChronicCond_Depression', 'ChronicCond_Diabetes',
       'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
       'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke',]

# %%
for i in categorical_cols[:3]:
    sns.countplot(x=i,data=final,hue='PotentialFraud',palette='BuPu',color='lightgreen',saturation=0.8)
    plt.show()

# %%
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22","#17becf"]

for i,col in zip(categorical_cols[4:],colors):
    sns.countplot(x='PotentialFraud',data=final,hue=i,palette='GnBu',order=['Yes','No'],color=col,edgecolor='black',saturation=0.8)
    plt.show()

# %%
# Calculating ages
 
#converting the object date into date format
final['DOB']=pd.to_datetime(final['DOB']).dt.year
final['age']=2009-final['DOB']


# %%
#Dropping DOB column
final.drop('DOB',axis=1,inplace=True)


# %%
age_sorted=sorted(map(int,final['age'].unique()))
print(*age_sorted)

# %%
plt.figure(figsize=(12,8))
sns.countplot(x="age", hue="PotentialFraud", data=final)
plt.xticks(rotation=90)
plt.show()

# %%
plt.figure(figsize=(12,8))
sns.kdeplot(final[final['PotentialFraud'] == 'Yes']['age'], label='Fraud',shade=True)
sns.kdeplot(final[final['PotentialFraud'] == 'No']['age'], label='No Fraud',color='Red')
plt.legend()
plt.show()


# %%
# calculating claim duration (days between claim start data and end date)
final['ClaimStartDt']=pd.to_datetime(final['ClaimStartDt'])
final['ClaimEndDt']=pd.to_datetime(final['ClaimEndDt'])
final['clm_duration']=(final['ClaimEndDt']-final['ClaimStartDt']).dt.days 

# %%
final.drop(['ClaimStartDt','ClaimEndDt'],axis=1,inplace=True)

# %%
days=final['clm_duration'].unique()
print(*days)

# %%
plt.figure(figsize=(12,6))
sns.countplot(x='clm_duration',data=final,hue='PotentialFraud')
plt.plot()

# %%
plt.figure(figsize=(12,6))
sns.kdeplot(final[final['PotentialFraud'] == 'Yes']['clm_duration'], label='Fraud',shade=True)
sns.kdeplot(final[final['PotentialFraud'] == 'No']['clm_duration'], label='No Fraud',color='Red')
plt.legend()
plt.show()

# %%
# Calculating Length Of Stay(LOS)
# two columns admission date and discharge date are present only in Inpatient data 
# finding the LOS in INpatient data and assign 0 to Outpatient data

final['AdmissionDt']=pd.to_datetime(final['AdmissionDt'],errors='coerce') 
final['DischargeDt']=pd.to_datetime(final['DischargeDt'],errors='coerce')  
                            # coerce - used to convert not parsable date format into NAT(not a date)

final['LOS']=(final['DischargeDt']-final['AdmissionDt']).dt.days

final['LOS'].fillna(0,inplace=True)


# %%
los_days=final['LOS'].unique()
print(*los_days)

# %%
plt.figure(figsize=(16,6))
sns.countplot(x='LOS',data=final,hue='PotentialFraud')
plt.plot()

# %%
plt.figure(figsize=(12,6))
sns.kdeplot(final[final['PotentialFraud'] == 'Yes']['LOS'], label='Fraud',shade=True)
sns.kdeplot(final[final['PotentialFraud'] == 'No']['LOS'], label='No Fraud',color='Red')
plt.legend()
plt.show()

# %%
# dropping admission date and discharge date 
final.drop(['AdmissionDt','DischargeDt'],axis=1,inplace=True)

# %%
# splitting the training data and testing data 
training_data=final[final['PotentialFraud'].notna()]    # OR x_train = final.dropna()
training_data

# %%
testing_data=final[final['PotentialFraud'].isna()]
testing_data

# %%
                                       #    Encoding

# %%

# frequency encoding,label encoding - doesn't consider the relation between column and target
# hash encoding - repetition of value possibility (-1,0,1)
# one-hot, get dummies - too many columns
# best target encoding but data leakage for single dataset
# Provider, BeneID, AttendingPhysician 

# %%
from category_encoders import TargetEncoder
encoder=TargetEncoder()
encoding_cols=['Provider','BeneID','AttendingPhysician','ClmDiagnosisCode_1','DiagnosisGroupCode']
training_data[encoding_cols]=encoder.fit_transform(training_data[encoding_cols],training_data['PotentialFraud'])
testing_data[encoding_cols]=encoder.transform(testing_data[encoding_cols])
training_data


# %%
# splitting the data into x_train and y_train  and making test data
x_train=training_data.drop(['PotentialFraud','ClaimID'],axis=1)
y_train=training_data['PotentialFraud']
x_test=testing_data.drop(['PotentialFraud','ClaimID'],axis=1)
x_train

# %%
#                                           Balancing 

# %%
from collections import Counter

print("Before SMOTE:", Counter(y_train))


# %%
# Find max allowed ratio
max_ratio = len(y_train[y_train == "Yes"]) / len(y_train[y_train == "No"])
print("Max Allowed Sampling Ratio:",max_ratio)
# print(f"Max Allowed Sampling Ratio: {max_ratio:.2f}")


# %%
x_train.dtypes

# %%
# Potential fraud  Yes = 38%  No = 62%   Imbalanced data   (471,650 No) and (221,953 Yes)
# using SMOTE (Synthetic Minority Oversampling Technique)
# from imblearn.combine import SMOTETomek
# smote=SMOTETomek(sampling_strategy=0.6,random_state=42)
# x_sampled,y_sampled=smote.fit_resample(x_train,y_train)

from imblearn.over_sampling import SMOTE
smote=SMOTE(sampling_strategy=0.62,random_state=42)
x_smote,y_smote=smote.fit_resample(x_train,y_train)

# %%
# y_smote

# %%
from imblearn.under_sampling import TomekLinks
tomek=TomekLinks()
x_final,y_final=tomek.fit_resample(x_smote,y_smote)


# %%
y_final

# %%
# need to encode or convert the categorical label column of y_train into numeric 
# coz xgboost not work with categorical column

# Mapping manually
y_fin=y_final
# y_final = y_final.map({'No': 0, 'Yes': 1})
y_final = y_final.map({'No':0,'Yes':1})
y_final

# %%
from sklearn.model_selection import train_test_split
a_train, a_test, b_train, b_test = train_test_split(x_final, y_final, test_size=0.2, random_state=1)

a_train

# %%
b_train

# %%
                            # Model Training 
# using Decision tree and Random Forest                             

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

Dt_entropy=DecisionTreeClassifier(criterion='entropy',random_state=42,max_depth=5)
Dt_gini=DecisionTreeClassifier(criterion='gini',random_state=42,max_depth=5)
rf=RandomForestClassifier(n_estimators=100,max_depth=10)
xg = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, scale_pos_weight=10)

models=[Dt_entropy,Dt_gini,rf,xg]
model_name=["DecisionTreeClassifier(Entropy)","DecisionTreeClassifier(Gini)","RandomForest","XGBoost"]

# Dt_entropy.fit(x_training,y_training)
# y_pred_DTE=Dt_entropy.predict(x_testing)
# y_pred_DTE


# %%
# Dt_gini.fit(x_final,y_final)
# y_pred_DTG=Dt_gini.predict(x_test)
# y_pred_DTG

# %%
# rf.fit(x_final,y_final)
# y_pred_rf=rf.predict(x_test)

# %%
from sklearn.metrics import accuracy_score,confusion_matrix,recall_score,f1_score,classification_report,precision_score,ConfusionMatrixDisplay

# %%
for i,j in zip(models,model_name):
    i.fit(a_train,b_train)
    b_pred=i.predict(a_test)
    print("***********",j,"****************")
    print('Accuracy_score = ',accuracy_score(b_test,b_pred))
    print('recall = ',recall_score(b_test,b_pred))
    print("Precision = ",precision_score(b_test,b_pred))
    print("Classification Report \n",classification_report(b_test,b_pred))
    

# %%
best_model =XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, scale_pos_weight=10)
best_model.fit(a_train,b_train)


# %%
import joblib

# Save the trained model
joblib.dump(best_model, 'Healthcare_fraud_detection_lessed.pkl')


# %%
# Load the model
best_model = joblib.load('Healthcare_fraud_detection_lessed.pkl')

# Make predictions using the loaded model
predictions = best_model.predict(a_test)  # Example with test data
predictions


# %%
a_train.columns

# %%
import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model and other artifacts
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Healthcare Fraud Detection")

# Input form
with st.form(key='form'):
    age = st.slider("Age", 0, 100, 45)
    claim_duration = st.slider("Claim Duration (in days)", 0, 365, 10)
    num_inpatient = st.number_input("Number of Inpatient Claims", min_value=0)
    num_outpatient = st.number_input("Number of Outpatient Claims", min_value=0)
    total_claim_amount = st.number_input("Total Claim Amount", min_value=0.0, format="%.2f")
    
    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    # Create input array
    input_data = pd.DataFrame([[age, claim_duration, num_inpatient, num_outpatient, total_claim_amount]],
                              columns=["Age", "ClaimDuration", "InpatientClaims", "OutpatientClaims", "TotalClaimAmount"])
    
    # Scale if needed
    input_data_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_data_scaled)[0]
    prob = model.predict_proba(input_data_scaled)[0][1]

    result = "Fraudulent" if prediction == 1 else "Non-Fraudulent"
    st.subheader(f"Prediction: {result}")
    st.write(f"Fraud Probability: {prob:.2f}")



