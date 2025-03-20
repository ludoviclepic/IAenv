import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableNCC(nn.Module):
    def __init__(self, num_classes, series_length, num_bands, embedding_dim=64):
        """
        Nearest Centroid Classifier with learnable prototypes and input-dependent time warping.
        """
        super(DeformableNCC, self).__init__()
        self.num_classes = num_classes
        self.series_length = series_length
        self.num_bands = num_bands
        # Prototypes: one learnable multivariate time series per class
        # Initialize prototypes as zeros or small random (could also use training data centroids for better start)
        self.prototypes = nn.Parameter(torch.zeros(num_classes, series_length, num_bands))
        
        # Encoder: reduce input time series to a latent feature (here a simple 1D CNN or LSTM could be used; we'll use a small CNN)
        self.encoder = nn.Sequential(
            nn.Conv1d(num_bands, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(10),  # compress time dimension to length 10
            nn.Flatten(),
            nn.Linear(32*10, embedding_dim), nn.ReLU()
        )
        # One warp and offset predictor per class (each takes the embedding and outputs a shift and offset)
        self.warp_predictors = nn.ModuleList([nn.Linear(embedding_dim, 1) for _ in range(num_classes)])
        self.offset_predictors = nn.ModuleList([nn.Linear(embedding_dim, 1) for _ in range(num_classes)])
    
    def forward(self, x):
        """
        Forward pass: Given input series x (batch_size x T x C), 
        return reconstructed series for each class prototype and the alignment parameters.
        """
        batch_size, T, C = x.shape
        # Encode input
        # Note: reshape to (batch, C, T) for Conv1d
        embed = self.encoder(x.permute(0, 2, 1))  # shape: (batch, embedding_dim)
        # For each class, predict warp (time shift) and offset
        warp_params = []   # list of tensors (batch x 1) for each class
        offset_params = []
        for k in range(self.num_classes):
            w = self.warp_predictors[k](embed)    # (batch, 1)
            o = self.offset_predictors[k](embed)  # (batch, 1)
            warp_params.append(w)
            offset_params.append(o)
        # Stack to get shape (batch, num_classes, 1)
        warp_params = torch.stack(warp_params, dim=1)  # (batch, K, 1)
        offset_params = torch.stack(offset_params, dim=1)  # (batch, K, 1)
        
        # Apply transformations to prototypes
        # Expand prototypes to batch: shape (batch, num_classes, T, C)
        prototypes = self.prototypes.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # Apply offset: add offset_param (broadcast over time and bands)
        aligned = prototypes + offset_params.unsqueeze(-1)  # broadcast offset over time steps
        
        # Apply time warping (shift) for each prototype:
        # We'll implement a simple linear interpolation for fractional shift.
        aligned_warped = aligned.clone()
        # Create a base index grid for time [0, T-1]
        base_idx = torch.arange(T, device=x.device).float()
        for k in range(self.num_classes):
            # shift each series in the batch by warp_params[:, k]
            # warp_params is in arbitrary units; assume output is fraction of series_length for simplicity
            shift = warp_params[:, k, 0]  # (batch,)
            # For each sample in batch, interpolate the prototype k
            for b in range(batch_size):
                idx_shifted = base_idx - shift[b]  # shifted index positions
                # clamp indices to [0, T-1]
                idx_clamped = torch.clamp(idx_shifted, 0, T-1)
                idx0 = idx_clamped.floor().long()
                idx1 = torch.clamp(idx0 + 1, 0, T-1)
                frac = idx_clamped - idx0.float()
                # linear interpolation between prototype[k] at idx0 and idx1
                prot = aligned[b, k]  # shape (T, C) for prototype k, sample b
                aligned_warped[b, k] = prot[idx0] * (1 - frac.unsqueeze(-1)) + prot[idx1] * (frac.unsqueeze(-1))
        return aligned_warped, warp_params, offset_params

def training_loop(model, train_loader, num_epochs=20, lr=1e-3, mu=1.0, nu=0.01):
    """
    Train the DeformableNCC model using reconstruction loss + contrastive loss + TV regularization.
    train_loader is an iterable of (x, y) batches.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(torch.float32)
            # Forward pass: get reconstructed series for all class prototypes
            recon_all, warp_params, offset_params = model(x)  # recon_all: (batch, K, T, C)
            batch_size, K, T, C = recon_all.shape
            # Compute reconstruction error for each class prototype
            # MSE per sample per class:
            # shape (batch, K): mean squared error between x and recon_all[:,k]
            mse = ((recon_all - x.unsqueeze(1))**2).mean(dim=(2,3))
            # Reconstruction loss (supervised): only for the true class of each sample
            # Gather the MSE of the correct class for each sample
            mse_true = mse[range(batch_size), y]
            L_rec = mse_true.mean()
            # Contrastive loss: encourages the true prototype to have smallest error
            # We'll use a cross-entropy formulation: p(k|x) = exp(-mse_k) / sum_j exp(-mse_j)
            # and maximize log p(y|x) (equivalently minimize negative log-likelihood)
            # Compute softmax over negative errors
            logits = -mse  # higher logit for smaller error
            labels = y.to(torch.long)
            L_cont = F.cross_entropy(logits, labels)
            # Total variation regularization on prototypes: encourage smooth temporal changes
            # TV = sum_k sum_c sum_{t=1}^{T-1} (P[k,t,c] - P[k,t-1,c])^2
            P = model.prototypes  # (K, T, C)
            tv = (P[:, 1:] - P[:, :-1]).pow(2).mean()  # mean over time and bands (sum could also be used)
            # Total loss
            loss = L_rec + mu * tv + nu * L_cont
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training loss: {avg_loss:.4f}")
