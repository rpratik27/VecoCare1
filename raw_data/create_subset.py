import pandas as pd

# Load the datasets
admissions = pd.read_csv("/Users/pratikranjan/Desktop/vecocare_v2.0/data/mimic3/raw/ADMISSIONS.csv")
noteevents = pd.read_csv("/Users/pratikranjan/Desktop/vecocare_v2.0/data/mimic3/raw/NOTEEVENTS.csv")
diagnoses = pd.read_csv("/Users/pratikranjan/Desktop/vecocare_v2.0/data/mimic3/raw/DIAGNOSES_ICD.csv")

# Get 100 unique patients from ADMISSIONS
selected_patients = admissions["SUBJECT_ID"].drop_duplicates().sample(n=100, random_state=42)

# Filter ADMISSIONS, NOTEEVENTS, and DIAGNOSES_ICD for these patients
subset_admissions = admissions[admissions["SUBJECT_ID"].isin(selected_patients)]
subset_noteevents = noteevents[noteevents["SUBJECT_ID"].isin(selected_patients)]
subset_diagnoses = diagnoses[diagnoses["SUBJECT_ID"].isin(selected_patients)]

# Save the subsets
subset_admissions.to_csv("ADMISSIONS_subset.csv", index=False)
subset_noteevents.to_csv("NOTEEVENTS_subset.csv", index=False)
subset_diagnoses.to_csv("DIAGNOSES_ICD_subset.csv", index=False)

print("Subset of 100 patients from ADMISSIONS, NOTEEVENTS, and DIAGNOSES_ICD saved!")
