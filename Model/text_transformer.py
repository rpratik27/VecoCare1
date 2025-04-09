import torch
import torch.nn as nn
import pickle

class TransformerEncoder(nn.Module):
    def __init__(self, d, num_layers=2, num_heads=4, dim_feedforward=256):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, note_vectors):
        return self.transformer_encoder(note_vectors)

# Load the base-encoded patient notes
with open("/Users/pratikranjan/Desktop/vecocare_v2.0/base_encoded_notes.pkl", "rb") as f:
    encoded_patient_notes = pickle.load(f)

# Transformer model
d = 128  # Embedding dimension from base encoding
transformer_model = TransformerEncoder(d)

# Process each patient separately
encoded_transformed_notes = {}
for pid, note_vector in encoded_patient_notes.items():
    note_tensor = torch.tensor(note_vector, dtype=torch.float32)  # Convert to tensor
    transformed_notes = transformer_model(note_tensor.unsqueeze(0)).squeeze(0)  # Process with Transformer
    encoded_transformed_notes[pid] = transformed_notes.detach().numpy()

# Save the transformed notes
with open("transformed_patient_notes.pkl", "wb") as f:
    pickle.dump(encoded_transformed_notes, f)
    

print("Transformed patient notes saved successfully.")
