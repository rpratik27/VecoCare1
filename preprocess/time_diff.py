import pandas as pd
import pickle

# Load ADMISSIONS table
admissions = pd.read_csv("/Users/pratikranjan/Desktop/vecocare_v2.0/subset_data/ADMISSIONS_subset.csv", usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME"])

# Convert ADMITTIME to datetime format
admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"])

# Sort by SUBJECT_ID and ADMITTIME
admissions = admissions.sort_values(by=["SUBJECT_ID", "ADMITTIME"])

# Compute time differences between consecutive admissions for each patient
time_diffs = {}
for subject_id, group in admissions.groupby("SUBJECT_ID"):
    # Compute differences in days between consecutive admissions
    diffs = group["ADMITTIME"].diff().dt.total_seconds() / (60 * 60 * 24)
    # Prepend a zero (for the first admission) and exclude the first NaN
    time_diffs[subject_id] = diffs.iloc[1:].tolist()

# Save as Pickle
with open("time_diffs.pkl", "wb") as f:
    pickle.dump(time_diffs, f)

print("Admission time differences saved as Pickle.")
