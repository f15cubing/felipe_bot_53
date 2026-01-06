import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split

# 1. Define the Neural Network
class NeuroChessNet(nn.Module):
    def __init__(self):
        super(NeuroChessNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(778, 256), # Increased width for 100k data
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)

# 2. Load and Prepare Data
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    data = np.load("chess_data3.npz")
    X = torch.tensor(data['x'], dtype=torch.float32)
    # Normalize: Convert centipawns to "pawn units" (divide by 100)
    y_raw = torch.tensor(data['y'], dtype=torch.float32).view(-1, 1)
    y = y_raw / 1000.0

    dataset = TensorDataset(X, y)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_subset, val_subset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_subset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=512)

    # 3. Initialize Model, Loss, and Optimizer
    model = NeuroChessNet().to(device)
    criterion = nn.HuberLoss(delta=0.1)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    epochs = 25
    best_val_loss = float('inf')

    # 4. The Training Loop
    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for v_X, v_y in val_loader:
                v_X, v_y = v_X.to(device), v_y.to(device)
                v_out = model(v_X)
                val_loss += criterion(v_out, v_y).item()
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "model.pth")
            print("--> Model Saved (Improved Validation)")

if __name__ == "__main__":
    train_model()
