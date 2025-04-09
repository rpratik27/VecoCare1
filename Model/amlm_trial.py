import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle

def apply_mask(Z0, MASK):
    """Applies masking to Z0 based on MASK indices."""
    for i in range(MASK.shape[0]):
        patient_idx, visit_idx, feature_idx = MASK[i]
        Z0[patient_idx, visit_idx, feature_idx] = 0
    return Z0

class TransformerWithConv(nn.Module):
    def __init__(self, d, num_heads, L):
        """Initialize the Transformer with convolution module."""
        super().__init__()
        self.L = L
        self.mhsa_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=d, num_heads=num_heads, batch_first=True) 
            for _ in range(L)
        ])
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=d, out_channels=d, kernel_size=3, padding=1) 
            for _ in range(L)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d)) 
            for _ in range(L)
        ])
        self.layer_norms_1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
        self.layer_norms_2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])

    def forward(self, Z0):
        Zl = Z0
        for l in range(self.L):
            Zl_norm = self.layer_norms_1[l](Zl)
            attn_output, _ = self.mhsa_layers[l](Zl_norm, Zl_norm, Zl_norm)
            conv_output = self.conv_layers[l](Zl_norm.permute(0, 2, 1)).permute(0, 2, 1)
            Zl = Zl + attn_output + conv_output
            Zl_norm = self.layer_norms_2[l](Zl)
            Zl = Zl + self.ffn_layers[l](Zl_norm)
        return Zl

def transformer_with_conv(Z0, MASK, num_heads=8, L=4):
    d = Z0.shape[-1]
    Z0 = apply_mask(Z0, MASK)
    model = TransformerWithConv(d=d, num_heads=num_heads, L=L)
    ZL = model(Z0)
    return ZL

def mlm_loss_patient(H, U, mask_ratio=0.15, num_heads=8, L=4):
    """
    Computes the AMLM loss for a single patient.
    H and U are tensors of shape (t, d) (time steps x feature dimension).
    This function ensures that H and U have matching time steps and feature dimensions.
    """
    # Get original shapes
    t_H, d_H = H.shape
    t_U, d_U = U.shape

    # Ensure both H and U have the same number of time steps by padding the shorter one.
    if t_H != t_U:
        new_t = max(t_H, t_U)
        if t_H < new_t:
            pad_tensor = torch.zeros(new_t - t_H, d_H, device=H.device)
            H = torch.cat([H, pad_tensor], dim=0)
        if t_U < new_t:
            pad_tensor = torch.zeros(new_t - t_U, d_U, device=U.device)
            U = torch.cat([U, pad_tensor], dim=0)

    # Ensure both H and U have the same feature dimension by padding the one with smaller dimension.
    if d_H != d_U:
        new_d = max(d_H, d_U)
        if d_H < new_d:
            pad_tensor = torch.zeros(H.shape[0], new_d - d_H, device=H.device)
            H = torch.cat([H, pad_tensor], dim=-1)
        if d_U < new_d:
            pad_tensor = torch.zeros(U.shape[0], new_d - d_U, device=U.device)
            U = torch.cat([U, pad_tensor], dim=-1)
        d = new_d
    else:
        d = d_H

    # Add batch dimension: shape becomes (1, t, d)
    H = H.unsqueeze(0)
    U = U.unsqueeze(0)
    # Concatenate H and U along feature dimension -> shape: (1, t, 2*d)
    Z0 = torch.cat((H, U), dim=-1)
    
    # Generate mask indices for this patient.
    # The mask is generated for the entire Z0 with shape (1, t, 2*d)
    _, t, two_d = Z0.shape
    mask_indices = torch.rand(1, t, two_d) < mask_ratio
    MASK = torch.nonzero(mask_indices, as_tuple=False)
    
    # Process through transformer with conv layers.
    ZL = transformer_with_conv(Z0, MASK, num_heads, L)
    
    # Split the output back into H' and U'
    H_prime, U_prime = torch.split(ZL, [d, d], dim=-1)
    # Create masks for each half
    C_mask = mask_indices[..., :d]
    N_mask = mask_indices[..., d:]
    Lc = F.mse_loss(H_prime[C_mask], H[C_mask])
    Ln = F.mse_loss(U_prime[N_mask], U[N_mask])
    return Lc, Ln

def load_data_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

# Load the base-encoded vectors (dictionaries mapping patient IDs to 2D arrays)
H_dict = load_data_from_pkl('/Users/pratikranjan/Desktop/vecocare_v2.0/base_encoded_visits.pkl')
U_dict = load_data_from_pkl('/Users/pratikranjan/Desktop/vecocare_v2.0/base_encoded_notes.pkl')

# Compute the loss for each patient separately
losses = {}
for pid in H_dict:
    H_patient = torch.tensor(H_dict[pid], dtype=torch.float32)
    U_patient = torch.tensor(U_dict[pid], dtype=torch.float32)
    Lc, Ln = mlm_loss_patient(H_patient, U_patient, mask_ratio=0.15, num_heads=8, L=4)
    losses[pid] = {'Lc': Lc.item(), 'Ln': Ln.item()}
    print(f"Patient {pid}: Lc = {Lc.item()}, Ln = {Ln.item()}")

# Optionally, aggregate the losses
avg_Lc = sum(loss['Lc'] for loss in losses.values()) / len(losses)
avg_Ln = sum(loss['Ln'] for loss in losses.values()) / len(losses)
print(f"Average Lc Loss: {avg_Lc}")
print(f"Average Ln Loss: {avg_Ln}")
