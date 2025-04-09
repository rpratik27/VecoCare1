import pandas as pd
import pickle

# File paths (update these if needed)
diagnoses_file = "/Users/pratikranjan/Desktop/vecocare_v2.0/subset_data/DIAGNOSES_ICD_subset.csv"
icd9_ccs_file = "/Users/pratikranjan/Downloads/ICD9PCS_CCS_CCSwebsite_2015_vid.csv"

# Load DIAGNOSES_ICD_subset.csv (ICD-9 codes)
df_diag = pd.read_csv(diagnoses_file, usecols=["ICD9_CODE"], dtype=str)

# Load ICD9PCS_CCS_CCSwebsite_2015.csv (ICD-9 to CCS mapping)
df_ccs = pd.read_csv(icd9_ccs_file, dtype=str)

# Function to normalize ICD-9 codes (remove leading zeros and decimal points)
def normalize_icd9(code):
    if isinstance(code, str):
        return code.replace(".", "").lstrip("0")  # Remove decimal and leading zeros
    return code

# Normalize ICD-9 codes in both datasets
df_diag["ICD9_CODE"] = df_diag["ICD9_CODE"].apply(normalize_icd9)
df_ccs["ICD-9-CM CODE"] = df_ccs["ICD-9-CM CODE"].apply(normalize_icd9)

# Convert ICD-9 to CCS mapping into a dictionary
icd9_to_ccs_map = dict(zip(df_ccs["ICD-9-CM CODE"], df_ccs["CCS CATEGORY"]))

# Create final ICD-9 to CCS mapping for diagnoses file
icd9_to_ccs = {code: icd9_to_ccs_map.get(code, "Unknown") for code in df_diag["ICD9_CODE"].unique()}

# Count the number of unique CCS categories
unique_ccs_categories = set(icd9_to_ccs.values()) - {"Unknown"}  # Exclude "Unknown"
num_unique_categories = len(unique_ccs_categories)

# Save the mapping as a Pickle file
with open("icd9_to_ccs.pkl", "wb") as f:
    pickle.dump(icd9_to_ccs, f)

# Print results
print(f"Mapped {len(icd9_to_ccs)} ICD-9 codes to CCS categories.")
print(f"Number of unique CCS categories found: {num_unique_categories}")
