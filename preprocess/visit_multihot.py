import pandas as pd
import numpy as np
import pickle

# Load DIAGNOSES_ICD table (replace with actual file path)
df = pd.read_csv(
    "/Users/pratikranjan/Desktop/vecocare_v2.0/subset_data/DIAGNOSES_ICD_subset.csv",
    usecols=["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]
)

# Load code_map from Pickle
with open("code_map.pkl", "rb") as f:
    code_map = pickle.load(f)

# Group by patient and visit, and create multi-hot vectors
patient_visits = {}
for (subject_id, hadm_id), group in df.groupby(["SUBJECT_ID", "HADM_ID"]):
    visit_vector = np.zeros(len(code_map), dtype=int)
    for code in group["ICD9_CODE"].dropna():
        if code in code_map:
            visit_vector[code_map[code]] = 1

    if subject_id not in patient_visits:
        patient_visits[subject_id] = []
    patient_visits[subject_id].append(visit_vector)

# Convert visit lists to NumPy arrays per patient
for subject_id in patient_visits:
    patient_visits[subject_id] = np.array(patient_visits[subject_id])

# Remove patients with only one visit from patient_visits.
patient_visits = {pid: visits for pid, visits in patient_visits.items() if visits.shape[0] >= 2}

# Now split each patient's visits:
# Use the first T-1 visits as inputs and the T-th visit as label.
patient_inputs = {}
patient_labels = {}
for subject_id, visits in patient_visits.items():
    patient_inputs[subject_id] = visits[:-1]  # all visits except the last
    patient_labels[subject_id] = visits[-1]    # the last visit as the label

# Save patient inputs and labels to separate pickle files
with open("patient_inputs_multihot.pkl", "wb") as f:
    pickle.dump(patient_inputs, f)
with open("patient_labels_multihot.pkl", "wb") as f:
    pickle.dump(patient_labels, f)

print("Patient inputs (T-1 visits) and labels (T-th visit) saved as separate pickle files.")

# Optionally, to verify that files can be loaded:
with open("patient_inputs_multihot.pkl", "rb") as f:
    loaded_patient_inputs = pickle.load(f)
with open("patient_labels_multihot.pkl", "rb") as f:
    loaded_patient_labels = pickle.load(f)

print("Pickle files loaded successfully.")
