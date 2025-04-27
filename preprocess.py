import numpy as np
import pandas as pd
import joblib
from category_encoders import TargetEncoder

encoder = joblib.load('target_encoder.pkl')
def preprocess_input(df):

  encods = ['Provider', 'BeneID', 'AttendingPhysician', 'ClmDiagnosisCode_1']

  # Encode all categorical columns at once
  df[encods] = encoder.transform(df[encods])
  
  
  
  
  # Process DOB
  if 'DOB' in df.columns:
      df['DOB'] = pd.to_datetime(df['DOB'], errors='coerce')
      df['age'] = 2009 - df['DOB'].dt.year
  else:
      df['age'] = 0  # or df['age'] = some default value
  df.drop(columns=['DOB'], errors='ignore', inplace=True)
  
  # Process Claim dates
  if 'ClaimStartDt' in df.columns and 'ClaimEndDt' in df.columns:
      df['ClaimStartDt'] = pd.to_datetime(df['ClaimStartDt'], errors='coerce')
      df['ClaimEndDt'] = pd.to_datetime(df['ClaimEndDt'], errors='coerce')
      df['clm_duration'] = (df['ClaimEndDt'] - df['ClaimStartDt']).dt.days
  else:
      df['clm_duration'] = 0
  df.drop(columns=['ClaimStartDt', 'ClaimEndDt'], errors='ignore', inplace=True)
  
  # Process Admission and Discharge dates
  if 'AdmissionDt' in df.columns and 'DischargeDt' in df.columns:
      df['AdmissionDt'] = pd.to_datetime(df['AdmissionDt'], errors='coerce')
      df['DischargeDt'] = pd.to_datetime(df['DischargeDt'], errors='coerce')
      df['LOS'] = (df['DischargeDt'] - df['AdmissionDt']).dt.days
      df['LOS'].fillna(0, inplace=True)
  else:
      df['LOS'] = 0
  df.drop(columns=['AdmissionDt', 'DischargeDt','ClaimID'], errors='ignore', inplace=True)

  return df
