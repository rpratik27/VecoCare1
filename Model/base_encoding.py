import torch
import torch.nn as nn
import pickle
import numpy as np

# Define the Base Encoder model
class BaseEncoder(nn.Module):
    def __init__(self, icd_dim, note_dim, d):
        super(BaseEncoder, self).__init__()
        self.icd_encoder = nn.Linear(icd_dim, d)
        self.note_encoder = nn.Linear(note_dim, d)

    def forward(self, icd_multi_hot, note_multi_hot):
        icd_encoded = self.icd_encoder(icd_multi_hot)
        note_encoded = self.note_encoder(note_multi_hot)
        return icd_encoded, note_encoded

# Function to load pickle files
def load_pkl(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

# Function to encode patient data (takes separate inputs)
def encode_patient_data(patient_visits, patient_notes, model, device,vocab):
    encoded_visits_dict = {}
    encoded_notes_dict = {}
    
    for pid in patient_visits.keys():
        icd_matrix = patient_visits[pid]  # ICD multi-hot matrix
        note_matrix = patient_notes.get(pid, np.zeros((1, len(vocab))))  # Default to zero if missing

        icd_tensor = torch.tensor(icd_matrix, dtype=torch.float32).to(device)
        note_tensor = torch.tensor(note_matrix, dtype=torch.float32).to(device)

        icd_encoded = model.icd_encoder(icd_tensor)
        note_encoded = model.note_encoder(note_tensor)

        # Convert back to numpy and store
        encoded_visits_dict[pid] = icd_encoded.detach().cpu().numpy()
        encoded_notes_dict[pid] = note_encoded.detach().cpu().numpy()

    return encoded_visits_dict, encoded_notes_dict

# Main function
if __name__ == "__main__":
    # Load actual data
    patient_visits = load_pkl("/Users/pratikranjan/Desktop/vecocare_v2.0/patient_visits_multihot.pkl")
    patient_notes = load_pkl("/Users/pratikranjan/Desktop/vecocare_v2.0/patient_notes_multihot.pkl")  # Fixed file path
    code_map = load_pkl("/Users/pratikranjan/Desktop/vecocare_v2.0/code_map.pkl")
    vocab = load_pkl("/Users/pratikranjan/Desktop/vecocare_v2.0/vocabulary.pkl")

    # Define model parameters
    icd_dim = len(code_map)
    note_dim = len(vocab)
    d = 128

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize BaseEncoder model and move to device
    model = BaseEncoder(icd_dim, note_dim, d).to(device)

    # Encode patient data
    encoded_visits, encoded_notes = encode_patient_data(patient_visits, patient_notes, model, device,vocab)

    # Save encoded data
    with open("base_encoded_visits.pkl", "wb") as f:
        pickle.dump(encoded_visits, f)

    with open("base_encoded_notes.pkl", "wb") as f:
        pickle.dump(encoded_notes, f)

    print("✅ Encoded patient visits and notes saved successfully.")
