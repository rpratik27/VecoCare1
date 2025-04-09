import pandas as pd
import numpy as np
import pickle

# Load DIAGNOSES_ICD table (replace with actual file path)
df = pd.read_csv("/Users/pratikranjan/Desktop/vecocare_v2.0/subset_data/DIAGNOSES_ICD_subset.csv", usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"])

# Load code_map from Pickle
with open("code_map.pkl", "rb") as f:
    code_map = pickle.load(f)

# Group by patient and visit
patient_visits = {}
for (subject_id, hadm_id), group in df.groupby(["SUBJECT_ID", "HADM_ID"]):
    visit_vector = np.zeros(len(code_map), dtype=int)
    for code in group["ICD9_CODE"].dropna():
        if code in code_map:
            visit_vector[code_map[code]] = 1
    
    if subject_id not in patient_visits:
        patient_visits[subject_id] = []
    patient_visits[subject_id].append(visit_vector)

# Convert visit lists to NumPy arrays
for subject_id in patient_visits:
    patient_visits[subject_id] = np.array(patient_visits[subject_id])

# Save as Pickle (default format)
with open("patient_visits_multihot.pkl", "wb") as f:
    pickle.dump(patient_visits, f)

print("Multi-hot encoded visit sequences saved as Pickle.")

# Load the saved Pickle file
with open("patient_visits.pkl", "rb") as f:
    loaded_patient_visits = pickle.load(f)

print("Pickle file loaded successfully.")
