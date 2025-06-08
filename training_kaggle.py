import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# --- Kaggle download (run once) ---
# 1) pip install kaggle
# 2) place your Kaggle API token in ~/.kaggle/kaggle.json
# 3) Uncomment the block below to download & unzip the data

# from kaggle.api.kaggle_api_extended import KaggleApi
# api = KaggleApi()
# api.authenticate()
# api.dataset_download_files(
#     'sudalairajkumar/cryptocurrencypricehistory',
#     path='data/',
#     unzip=True
# )

# --- Load Bitcoin CSV ---
# Assumes CSV named like 'bitcoin_up_to_2021-03-31.csv'
csv_files = glob.glob('data/bitcoin*.csv')
if not csv_files:
    raise FileNotFoundError("No Bitcoin CSV found in data/; run the Kaggle download block first.")

df = pd.read_csv(csv_files[0], parse_dates=['Date'])
df.sort_values('Date', inplace=True)

# --- Feature selection & scaling ---
# Using only 'Close' price; you can add ['Open','High','Low','Volume']
features = df[['Close']].values  # shape (N,1)
scaler = MinMaxScaler()
features = scaler.fit_transform(features)

target = features  # for next-step prediction on 'Close'

# --- Sequence Dataset ---
class SequenceDataset(Dataset):
    def __init__(self, data, seq_len=60):
        self.seq_len = seq_len
        self.data = data

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        # Convert to float32
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )

# --- Prepare DataLoaders ---
seq_len = 60
batch_size = 32

dataset = SequenceDataset(features, seq_len)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_ds, batch_size=batch_size)

# --- Model import and instantiation ---
# Ensure transformer.py is in the same directory or in PYTHONPATH
from transformer import build_model, device

model = build_model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.MSELoss()

# --- Training loop ---
def train_one_epoch():
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output.squeeze(-1), y.squeeze(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(train_loader.dataset)

# --- Evaluation ---
def evaluate():
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output.squeeze(-1), y.squeeze(-1))
            total_loss += loss.item() * x.size(0)
    return total_loss / len(test_loader.dataset)

# --- Main training runner ---
if __name__ == "__main__":
    epochs = 20
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch()
        test_loss = evaluate()
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f}")

    # Example inference on last window
    model.eval()
    last_seq = torch.tensor(features[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(last_seq).cpu().item()
    print("Next-day scaled-close prediction:", pred)
    print("Unscaled:", scaler.inverse_transform([[pred]])[0][0])
