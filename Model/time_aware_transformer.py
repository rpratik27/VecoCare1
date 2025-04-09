import torch
import torch.nn as nn
import pickle
import math

class TimeAwareTransformer(nn.Module):
    def __init__(self, d_model, num_layers=2, num_heads=8, dim_feedforward=512):
        """
        d_model: final concatenated dimension (256 in your case)
        """
        super(TimeAwareTransformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, visit_vectors, time_differences):
        # If time_differences has shape [B, 1, 128] and visit_vectors has [B, num_visits, 128],
        # expand the second dimension of time_differences to match the number of visits.
        if time_differences.size(1) == 1 and visit_vectors.size(1) > 1:
            time_differences = time_differences.expand(-1, visit_vectors.size(1), -1)
        # Concatenate along the last dimension: final shape [B, num_visits, 256]
        combined_input = torch.cat((visit_vectors, time_differences), dim=-1)
        return self.transformer_encoder(combined_input)

# Load encoded visit vectors and time differences
with open("/Users/pratikranjan/Desktop/vecocare_v2.0/base_encoded_visits.pkl", "rb") as f:
    encoded_patient_visits = pickle.load(f)

with open("time_diffs.pkl", "rb") as f:
    time_diffs = pickle.load(f)

# We want the final concatenated dimension to be 256 (i.e. 128 from visit vector + 128 from time info)
transformer_model = TimeAwareTransformer(d_model=256, num_heads=8)

# Process each patient separately
encoded_transformed_visits = {}
for pid, visit_vector in encoded_patient_visits.items():
    # visit_tensor: expected shape (num_visits, 128)
    visit_tensor = torch.tensor(visit_vector, dtype=torch.float32)
    
    # Get time differences; original time vector is assumed to be shorter (one value per transition)
    # so we start with shape (num_visits - 1, 1)
    time_tensor = torch.tensor(time_diffs.get(pid, []), dtype=torch.float32).unsqueeze(1)
    
    # Add a 0 at the start for the first visit so that time_tensor has shape (num_visits, 1)
    time_tensor = torch.cat((torch.zeros(1, 1), time_tensor), dim=0)
    
    # Expand time_tensor from shape (num_visits, 1) to (num_visits, 128)
    # This repeats the same time difference value across 128 features.
    time_tensor = time_tensor.expand(-1, 128)
    
    # Add batch dimension so that:
    # visit_tensor becomes [1, num_visits, 128] and time_tensor becomes [1, num_visits, 128]
    visit_tensor = visit_tensor.unsqueeze(0)
    time_tensor = time_tensor.unsqueeze(0)
    
    # Process with the transformer
    transformed_visits = transformer_model(visit_tensor, time_tensor).squeeze(0)
    encoded_transformed_visits[pid] = transformed_visits.detach().numpy()

# Save the transformed visits
with open("transformed_patient_visits.pkl", "wb") as f:
    pickle.dump(encoded_transformed_visits, f)

print("Transformed patient visits saved successfully.")
