import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # progress bar

class DeformableNCC(nn.Module):
    def __init__(self, num_classes, series_length, num_bands, embedding_dim=64):
        super(DeformableNCC, self).__init__()
        self.num_classes = num_classes
        self.series_length = series_length
        self.num_bands = num_bands
        self.prototypes = nn.Parameter(torch.zeros(num_classes, series_length, num_bands))
        
        self.encoder = nn.Sequential(
            nn.Conv1d(num_bands, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(10),
            nn.Flatten(),
            nn.Linear(32*10, embedding_dim), nn.ReLU()
        )
        self.warp_predictors = nn.ModuleList([nn.Linear(embedding_dim, 1) for _ in range(num_classes)])
        self.offset_predictors = nn.ModuleList([nn.Linear(embedding_dim, 1) for _ in range(num_classes)])
    
    def forward(self, x):
        batch_size, T, C = x.shape
        embed = self.encoder(x.permute(0, 2, 1))
        warp_params = []
        offset_params = []
        for k in range(self.num_classes):
            w = self.warp_predictors[k](embed)
            o = self.offset_predictors[k](embed)
            warp_params.append(w)
            offset_params.append(o)
        warp_params = torch.stack(warp_params, dim=1)
        offset_params = torch.stack(offset_params, dim=1)
        
        prototypes = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1, -1)
        aligned = prototypes + offset_params.unsqueeze(-1)
        aligned_warped = aligned.clone()
        base_idx = torch.arange(T, device=x.device).float()
        for k in range(self.num_classes):
            shift = warp_params[:, k, 0]
            for b in range(batch_size):
                idx_shifted = base_idx - shift[b]
                idx_clamped = torch.clamp(idx_shifted, 0, T-1)
                idx0 = idx_clamped.floor().long()
                idx1 = torch.clamp(idx0 + 1, 0, T-1)
                frac = idx_clamped - idx0.float()
                prot = aligned[b, k]
                aligned_warped[b, k] = prot[idx0] * (1 - frac.unsqueeze(-1)) + prot[idx1] * (frac.unsqueeze(-1))
        return aligned_warped, warp_params, offset_params

def training_loop(model, train_loader, num_epochs=20, lr=1e-3, mu=1.0, nu=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in tqdm(range(num_epochs), desc="Supervised Training Epochs"):
        model.train()
        total_loss = 0.0
        for x, y in tqdm(train_loader, desc="Batch", leave=False):
            x = x.to(torch.float32)
            recon_all, warp_params, offset_params = model(x)
            batch_size, K, T, C = recon_all.shape
            mse = ((recon_all - x.unsqueeze(1))**2).mean(dim=(2,3))
            mse_true = mse[range(batch_size), y]
            L_rec = mse_true.mean()
            logits = -mse
            labels = y.to(torch.long)
            L_cont = F.cross_entropy(logits, labels)
            P = model.prototypes
            tv = (P[:, 1:] - P[:, :-1]).pow(2).mean()
            loss = L_rec + mu * tv + nu * L_cont
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
        avg_loss = total_loss / len(train_loader.dataset)
        tqdm.write(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
