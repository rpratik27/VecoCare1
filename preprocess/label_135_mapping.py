import pickle

# Load necessary files
with open("patient_labels_multihot.pkl", "rb") as f:
    patient_icd9_multihot = pickle.load(f)

with open("code_map.pkl", "rb") as f:
    code_map = pickle.load(f)

with open("icd9_to_ccs.pkl", "rb") as f:
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

# Convert ICD-9 multi-hot vectors to CCS multi-hot vectors for each patient
patient_ccs_multihot = {}

for patient_id, icd9_vector in patient_icd9_multihot.items():
    ccs_vector = icd9_to_ccs_vector(icd9_vector, code_map, icd9_to_ccs, ccs_to_index)
    patient_ccs_multihot[patient_id] = ccs_vector

# Save the CCS multi-hot vectors
with open("patient_ccs_multihot.pkl", "wb") as f:
    pickle.dump(patient_ccs_multihot, f)

print(f"Converted {len(patient_ccs_multihot)} patients' ICD-9 vectors to CCS multi-hot vectors.")
print(f"Total unique CCS categories: {len(ccs_to_index)}")
