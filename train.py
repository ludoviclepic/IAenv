from pathlib import Path
from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import os  
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" 

from collate import pad_collate
from dataset import PASTIS_Dataset
from unet3d import UNet


def print_iou_per_class(targets: torch.Tensor, preds: torch.Tensor, nb_classes: int) -> None:
    iou_per_class = []
    for class_id in range(nb_classes):
        iou = jaccard_score(
            targets == class_id,
            preds == class_id,
            average="binary",
            zero_division=0,
        )
        iou_per_class.append(iou)

    for class_id, iou in enumerate(iou_per_class):
        print(
            f"class {class_id} - IoU: {iou:.4f} - targets: {(targets == class_id).sum()} - preds: {(preds == class_id).sum()}"
        )


def print_mean_iou(targets: torch.Tensor, preds: torch.Tensor) -> None:
    mean_iou = jaccard_score(targets, preds, average="macro")
    print(f"meanIOU (over existing classes in targets): {mean_iou:.4f}")


def train_model(
    data_folder: Path,
    nb_classes: int,
    input_channels: int,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
) -> UNet:
    dataset = PASTIS_Dataset(data_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=pad_collate, shuffle=True
    )

    model = UNet(input_channels, nb_classes, dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device(device)
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data_dict, date_dict = inputs
            data_dict["S2"] = data_dict["S2"].to(device)
            targets = targets.to(device)

            # Fix target shape
            if targets.dim() == 2:
                targets = targets.unsqueeze(0)  # (H, W) → (1, H, W)
            elif targets.dim() == 3 and targets.shape[1] != data_dict["S2"].shape[-2]:
                targets = targets.unsqueeze(-1)  # (B, H) → (B, H, 1)

            # Final check
            if targets.dim() != 3:
                raise ValueError(f"targets must be 3D (B, H, W), got {targets.shape}")

            optimizer.zero_grad()

            outputs = model(data_dict["S2"])
            outputs_median_time = torch.median(outputs, dim=2).values  # (B, C, H, W)

            print("outputs_median_time.shape:", outputs_median_time.shape)
            print("targets.shape (before loss):", targets.shape)
            
            # Clamp target values to valid range to avoid out of bounds error
            targets = targets.clamp(0, nb_classes - 1)
            
            loss = criterion(outputs_median_time, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            preds = torch.argmax(outputs_median_time, dim=1)

            targets_np = targets.cpu().numpy().flatten()
            preds_np = preds.cpu().numpy().flatten()

            if verbose:
                print_iou_per_class(targets_np, preds_np, nb_classes)
                print_mean_iou(targets_np, preds_np)

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")
    return model


if __name__ == "__main__":
    model = train_model(
        data_folder=Path("/Users/ludoviclepic/Downloads/Pastis"),
        nb_classes=18,
        input_channels=10,
        num_epochs=10,
        batch_size=5,
        learning_rate=1e-3,
        device="cpu",
        verbose=True,
    )
