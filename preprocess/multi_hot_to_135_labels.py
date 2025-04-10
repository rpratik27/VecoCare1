import pickle

# Load the ICD-9 to CCS mapping
with open("/Users/pratikranjan/Desktop/vecocare_v2.0/icd9_to_ccs.pkl", "rb") as f:
    icd9_to_ccs = pickle.load(f)

# Extract unique CCS categories and create an index mapping
unique_ccs_categories = sorted(set(icd9_to_ccs.values()) - {"Unknown"})
ccs_to_index = {ccs: idx for idx, ccs in enumerate(unique_ccs_categories)}

# Function to convert ICD-9 multi-hot vector to CCS multi-hot vector
def icd9_to_ccs_vector(icd9_vector, code_map, icd9_to_ccs, ccs_to_index):
    """Convert ICD-9 multi-hot vector to CCS multi-hot vector."""
    ccs_vector = [0] * len(ccs_to_index)  # Initialize CCS multi-hot vector

    for icd9_code, idx in code_map.items():
        if icd9_vector[idx] == 1:  # If ICD-9 code is present
            ccs_category = icd9_to_ccs.get(icd9_code, "Unknown")
            if ccs_category in ccs_to_index:
                ccs_vector[ccs_to_index[ccs_category]] = 1  # Mark CCS category as present

    return ccs_vector

# Print number of CCS categories
print(f"Total CCS categories: {len(ccs_to_index)}")
