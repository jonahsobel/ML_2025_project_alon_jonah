import pandas as pd
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
from tqdm import tqdm

def compute_global_min_max(csv_path, col_name='Close', chunk_size=10000):
    """
    Compute global min and max of a column by streaming through CSV in chunks.
    
    Args:
        csv_path: Path to the CSV file
        col_name: Name of the column to analyze
        chunk_size: Number of rows to read per chunk
        
    Returns:
        tuple: (global_min, global_max)
    """
    global_min = float('inf')
    global_max = float('-inf')
    
    print(f"Computing global min/max for column '{col_name}' with chunk_size={chunk_size}...")
    
    chunk_count = 0
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if col_name not in chunk.columns:
            raise ValueError(f"Column '{col_name}' not found in CSV")
        
        chunk_min = chunk[col_name].min()
        chunk_max = chunk[col_name].max()
        
        global_min = min(global_min, chunk_min)
        global_max = max(global_max, chunk_max)
        
        chunk_count += 1
        if chunk_count % 100 == 0:
            print(f"Processed {chunk_count} chunks...")
    
    print(f"Global min: {global_min}, Global max: {global_max}")
    return global_min, global_max


class BTCIterableDataset(IterableDataset):
    """
    Iterable dataset that streams Close prices and yields (sequence, target) pairs.
    Maintains a rolling buffer to efficiently create sequential samples.
    """
    
    def __init__(self, csv_path, seq_len, chunk_size, col_name='Close'):
        self.csv_path = csv_path
        self.seq_len = seq_len
        self.col_name = col_name
        self.chunk_size = chunk_size
        
    def __iter__(self):
        buffer = deque(maxlen=self.seq_len + 1)
        
        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunk_size):
            if self.col_name not in chunk.columns:
                raise ValueError(f"Column '{self.col_name}' not found in CSV")
            
            prices = chunk[self.col_name].values
            
            for price in prices:
                buffer.append(float(price))
                
                # Once buffer is full, yield (sequence, target) pairs
                if len(buffer) == self.seq_len + 1:
                    sequence = list(buffer)[:-1]  # First seq_len elements
                    target = buffer[-1]           # Last element
                    yield (sequence, target)


def collate_global(batch, global_min, global_max):
    """
    Collate function that builds tensors and applies global min-max normalization.
    
    Args:
        batch: List of (sequence, target) tuples
        global_min: Global minimum value for normalization
        global_max: Global maximum value for normalization
        
    Returns:
        tuple: (normalized_sequences, normalized_targets) as tensors
    """
    sequences, targets = zip(*batch)
    
    # Convert to tensors
    sequences = torch.tensor(sequences, dtype=torch.float32)  # (B, seq_len)
    targets = torch.tensor(targets, dtype=torch.float32)      # (B,)
    
    # Apply global min-max normalization
    sequences = (sequences - global_min) / (global_max - global_min)
    targets = (targets - global_min) / (global_max - global_min)
    
    # Reshape for model: sequences -> (B, seq_len, 1), targets -> (B, 1)
    sequences = sequences.unsqueeze(-1)
    targets = targets.unsqueeze(-1)
    
    # Move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequences = sequences.to(device)
    targets = targets.to(device)
    
    return sequences, targets


def get_dataloader(
    csv_path,
    seq_len,
    chunk_size,
    batch_size,
    col_name='Close',
    global_min=3.8,
    global_max=111975.0,
    compute_globals=False
):
    """
    Create a DataLoader for BTC price prediction with streaming and normalization.
    
    Args:
        csv_path: Path to the CSV file
        seq_len: Length of input sequences
        chunk_size: Number of rows to read per chunk
        batch_size: Batch size for training
        col_name: Name of the price column
        global_min: Hard-coded global minimum for normalization
        global_max: Hard-coded global maximum for normalization
        compute_globals: If True, recompute global min/max from the file
    
    Returns:
        tuple: (dataloader, global_min, global_max)
    """
    # Compute global stats if requested
    if compute_globals:
        global_min, global_max = compute_global_min_max(csv_path, col_name, chunk_size)
    
    # Create dataset with chunk size
    dataset = BTCIterableDataset(csv_path, seq_len, chunk_size, col_name)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_global(batch, global_min, global_max)
    )
    
    return dataloader, global_min, global_max


def train_steps(model, dataloader, num_steps, learning_rate=1e-4, print_every=100):
    """
    Train the model for a fixed number of steps.
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"Starting training for {num_steps} steps...")
    
    loss_sum = 0.0
    data_iter = iter(dataloader)
    
    for _ in tqdm(range(num_steps)):
        try:
            sequences, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            sequences, targets = next(data_iter)
        
        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
    
    print("Training completed!")


def predict_next(
    model,
    csv_path,
    seq_len,
    chunk_size,
    global_min,
    global_max,
    col_name='Close'
):
    """
    Predict the next price using the last seq_len prices from the CSV.
    
    Args:
        model: Trained transformer model
        csv_path: Path to the CSV file
        seq_len: Length of input sequence
        chunk_size: Number of rows to read per chunk when fetching last prices
        global_min: Global minimum for denormalization
        global_max: Global maximum for denormalization
        col_name: Name of the price column
        
    Returns:
        float: Predicted next price (denormalized)
    """
    model.eval()
    
    # Read the full file in chunks to collect last values
    last_prices = []
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        if col_name not in chunk.columns:
            raise ValueError(f"Column '{col_name}' not found in CSV")
        last_prices.extend(chunk[col_name].values.tolist())
    
    if len(last_prices) < seq_len:
        raise ValueError(f"Not enough data: need {seq_len}, got {len(last_prices)}")
    
    # Take last seq_len prices
    sequence = last_prices[-seq_len:]
    
    # Normalize and reshape for model
    sequence = torch.tensor(sequence, dtype=torch.float32)
    sequence = (sequence - global_min) / (global_max - global_min)
    sequence = sequence.unsqueeze(0).unsqueeze(-1)  # -> (1, seq_len, 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequence = sequence.to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(sequence)
        prediction = prediction.cpu().item()
    
    # Denormalize
    predicted_price = prediction * (global_max - global_min) + global_min
    print(f"Predicted next price: ${predicted_price:.2f}")
    return predicted_price
