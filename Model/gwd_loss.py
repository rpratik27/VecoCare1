import torch
import torch.nn.functional as F
import pickle
from torch.nn.utils.rnn import pad_sequence

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def pairwise_cosine_similarity(X, eps=1e-8):
    norms = torch.norm(X, dim=1, keepdim=True)
    X_norm = X / (norms + eps)
    return torch.mm(X_norm, X_norm.t())

def ground_distance_torch(H, U, tau_g):
    n = H.shape[0]
    m = U.shape[0]
    C_H = torch.exp(-pairwise_cosine_similarity(H) / tau_g)
    C_U = torch.exp(-pairwise_cosine_similarity(U) / tau_g)
    S_H2 = torch.sum(C_H**2, dim=1)
    S_U2 = torch.sum(C_U**2, dim=1)
    S_H = torch.sum(C_H, dim=1)
    S_U = torch.sum(C_U, dim=1)
    M = S_H2.unsqueeze(1) * m + S_U2.unsqueeze(0) * n - 2 * (S_H.unsqueeze(1) * S_U.unsqueeze(0))
    return M

def optimal_transport_torch(ph, pu, C, reg=0.01, max_iter=100, epsilon=1e-8):
    K = torch.exp(-C / (reg + epsilon))
    u = torch.ones_like(ph) / ph.shape[0]
    v = torch.ones_like(pu) / pu.shape[0]
    for _ in range(max_iter):
        u = ph / (torch.matmul(K, v) + epsilon)
        v = pu / (torch.matmul(K.t(), u) + epsilon)
    T = torch.ger(u, v) * K
    return T

def compute_gwd_torch(H, U, tau_g, reg=0.01):
    n = H.shape[0]
    m = U.shape[0]
    ph = torch.ones(n, device=H.device) / n
    pu = torch.ones(m, device=U.device) / m
    C = ground_distance_torch(H, U, tau_g)
    T = optimal_transport_torch(ph, pu, C, reg=reg)
    gwd = torch.sum(T * C)
    return gwd

def compute_gwd_infonce_loss_torch(H_batch, U_batch, tau_g, tau, reg=0.01):
    B = H_batch.shape[0]
    losses_v2t = []
    losses_t2v = []
    for i in range(B):
        H_i = H_batch[i]  # (T, d)
        U_i = U_batch[i]  # (T, d)
        D_i = compute_gwd_torch(H_i, U_i, tau_g, reg)
        all_v2t = [compute_gwd_torch(H_i, U_batch[k], tau_g, reg) for k in range(B)]
        all_t2v = [compute_gwd_torch(U_i, H_batch[k], tau_g, reg) for k in range(B)]
        all_v2t = torch.stack(all_v2t)
        all_t2v = torch.stack(all_t2v)
        neg_v2t = torch.sum(torch.exp(-all_v2t / tau)) + 1e-8
        neg_t2v = torch.sum(torch.exp(-all_t2v / tau)) + 1e-8
        loss_v2t = -torch.log(torch.exp(-D_i / tau) / neg_v2t)
        loss_t2v = -torch.log(torch.exp(-D_i / tau) / neg_t2v)
        losses_v2t.append(loss_v2t)
        losses_t2v.append(loss_t2v)
    final_loss = (torch.sum(torch.stack(losses_v2t)) + torch.sum(torch.stack(losses_t2v))) / (2 * B)
    return final_loss

def main():
    # Load joint embeddings dictionary from pickle
    data = load_pickle("/Users/pratikranjan/Desktop/vecocare_v2.0/joint_embeddings.pkl")  # Update path as needed
    
    # Convert dictionary entries into lists of tensors
    H_list = []
    U_list = []
    for pid in data.keys():
        H_tensor = torch.tensor(data[pid]['visit_embedding'], dtype=torch.float32)
        U_tensor = torch.tensor(data[pid]['note_embedding'], dtype=torch.float32)
        H_list.append(H_tensor)
        U_list.append(U_tensor)
    
    # Pad sequences so that all patients have the same number of visits (T)
    # pad_sequence returns (B, T, d) if batch_first=True.
    H_batch = torch.nn.utils.rnn.pad_sequence(H_list, batch_first=True)
    U_batch = torch.nn.utils.rnn.pad_sequence(U_list, batch_first=True)
    
    # Set loss hyperparameters
    tau_g = 0.1
    tau = 0.05
    
    # Compute the differentiable GWD InfoNCE loss
    loss = compute_gwd_infonce_loss_torch(H_batch, U_batch, tau_g, tau)
    print("Differentiable GWD InfoNCE Loss:", loss.item())
    
    # Backpropagation step (for training)
    # loss.backward()
    print("Backpropagation complete.")
    
if __name__ == "__main__":
    main()
