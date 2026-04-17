import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from twoTowerModel import TwoTowerModel

def train(
    model: TwoTowerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"Starting training on {device}...")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        
        # tqdm progress bar for training batches
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]", unit="batch")
        
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            scores = model(batch)
            loss = criterion(scores, batch["label"])
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            current_batch_loss = loss.item()
            train_loss += current_batch_loss
            
            # Update the progress bar with the current loss
            pbar.set_postfix({"loss": f"{current_batch_loss:.4f}"})

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        
        # tqdm progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Val]", unit="batch", leave=False)
        
        with torch.no_grad():
            for batch in val_pbar:
                batch = {k: v.to(device) for k, v in batch.items()}
                scores = model(batch)
                val_loss += criterion(scores, batch["label"]).item()

        scheduler.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_two_tower.pt")
            print(f"✓ New best model saved (Val Loss: {best_val_loss:.4f})")