import pandas as pd
import json
import pickle

# Load DIAGNOSES_ICD table (replace with actual file path)
df = pd.read_csv("/Users/pratikranjan/Desktop/vecocare_v2.0/subset_data/DIAGNOSES_ICD_subset.csv", usecols=["ICD9_CODE"])

# Drop duplicates and sort unique ICD-9 codes
unique_codes = sorted(df["ICD9_CODE"].dropna().unique())

# Create a vocabulary mapping ICD-9 codes to unique indices
code_map = {code: idx for idx, code in enumerate(unique_codes)}

# Save as Pickle
with open("code_map.pkl", "wb") as f:
    pickle.dump(code_map, f)

print(f"ICD-9 vocabulary saved with {len(code_map)} unique codes.")
