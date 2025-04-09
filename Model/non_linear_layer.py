import torch
import torch.nn as nn
import pickle

# Define the joint embedding mapper with non-linear projection layers
class JointEmbeddingMapper(nn.Module):
    def __init__(self, d):
        super(JointEmbeddingMapper, self).__init__()
        # gv: projects from 2*d to d (for visits)
        self.gv = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.LayerNorm(d)
        )
        # gs: projects from d to d (for notes)
        self.gs = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.LayerNorm(d)
        )
        
    def forward(self, transformed_notes, transformed_visits):
        # transformed_notes: shape (batch, seq_len, d)
        # transformed_visits: shape (batch, seq_len, 2*d)
        note_embedding = self.gs(transformed_notes)
        visit_embedding = self.gv(transformed_visits)
        return note_embedding, visit_embedding

# Set the base dimension
d = 128

# Load transformed patient notes and visits from pickle files
with open("transformed_patient_notes.pkl", "rb") as f:
    transformed_patient_notes = pickle.load(f)
with open("transformed_patient_visits.pkl", "rb") as f:
    transformed_patient_visits = pickle.load(f)

# Initialize the mapper
mapper = JointEmbeddingMapper(d)
mapper.eval()  # set to evaluation mode

# Map to joint embedding space for each patient
joint_embeddings = {}
for pid in transformed_patient_notes:
    # Convert to torch tensors
    note_tensor = torch.tensor(transformed_patient_notes[pid], dtype=torch.float32)    # Expected shape: (seq_len, d)
    visit_tensor = torch.tensor(transformed_patient_visits[pid], dtype=torch.float32)  # Expected shape: (seq_len, 2*d)
    
    # Add a batch dimension
    note_tensor = note_tensor.unsqueeze(0)   # shape: (1, seq_len, d)
    visit_tensor = visit_tensor.unsqueeze(0) # shape: (1, seq_len, 2*d)
    
    # Apply projection layers
    note_emb, visit_emb = mapper(note_tensor, visit_tensor)
    
    # Remove batch dimension and store as numpy arrays
    joint_embeddings[pid] = {
        'note_embedding': note_emb.squeeze(0).detach().numpy(),
        'visit_embedding': visit_emb.squeeze(0).detach().numpy()
    }

# Save the joint embeddings to a pickle file
with open("joint_embeddings.pkl", "wb") as f:
    pickle.dump(joint_embeddings, f)

print("Joint embeddings saved successfully.")
