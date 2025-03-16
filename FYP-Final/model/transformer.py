import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to input
        return x + self.pe[:, :x.size(1), :]

class EHRImputationTransformer(nn.Module):
    def __init__(self, feature_dim, d_model=128, nhead=8, num_layers=4, dropout=0.1):
        super(EHRImputationTransformer, self).__init__()
        
        # Input embedding layer
        self.embedding = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Missing value mask embedding (0 for observed, 1 for missing)
        self.mask_embedding = nn.Embedding(2, d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=4*d_model, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, missing_mask):
        """
        Args:
            x: tensor of shape [batch_size, seq_len, feature_dim]
               contains the original data with zeros in place of missing values
            missing_mask: tensor of shape [batch_size, seq_len, feature_dim]
               binary mask where 1 indicates missing values
        """
        batch_size, seq_len, feature_dim = x.shape
        
        # Create a binary mask for attention (1 for valid positions, 0 for padding)
        # Here we assume all sequences are valid (no padding)
        attention_mask = torch.ones(batch_size, seq_len, device=x.device)
        
        # Flatten the feature dimension for embedding
        x_flat = x.reshape(batch_size, seq_len * feature_dim)
        missing_mask_flat = missing_mask.reshape(batch_size, seq_len * feature_dim)
        
        # Embed the input
        x_emb = self.embedding(x_flat.view(-1, feature_dim)).view(batch_size, seq_len, -1)
        
        # Apply positional encoding
        x_emb = self.positional_encoding(x_emb)
        
        # Add missing value mask embedding
        mask_emb = self.mask_embedding(missing_mask_flat.long().view(-1, feature_dim).to(torch.int)).view(batch_size, seq_len, -1)
        x_emb = x_emb + mask_emb
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x_emb)
        
        # Project back to original feature space
        imputed_values = self.output_projection(transformer_out.reshape(-1, transformer_out.size(-1)))
        imputed_values = imputed_values.view(batch_size, seq_len, feature_dim)
        
        # Only replace missing values
        result = x * (1 - missing_mask) + imputed_values * missing_mask
        
        return result, imputed_values

class EHRDataset(torch.utils.data.Dataset):
    def __init__(self, data, mask=None):
        """
        Args:
            data: Original data tensor [num_samples, seq_len, feature_dim]
            mask: Missing value mask [num_samples, seq_len, feature_dim]
                 where 1 indicates missing values
        """
        self.data = data
        
        # If no mask is provided, create one (for training with artificial missingness)
        if mask is None:
            # Create random mask for training (this can be adjusted based on your needs)
            self.mask = torch.bernoulli(torch.ones_like(data) * 0.2)
        else:
            self.mask = mask
            
        # Create input data by zeroing out missing values
        self.input_data = self.data * (1 - self.mask)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.mask[idx], self.data[idx]

def train_imputation_model(model, train_loader, val_loader, epochs=50, lr=0.001, device='cuda'):
    """Train the imputation model"""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (x, mask, target) in enumerate(train_loader):
            x, mask, target = x.to(device), mask.to(device), target.to(device)
            
            optimizer.zero_grad()
            output, _ = model(x, mask)
            
            # Compute loss only on missing values
            loss = F.mse_loss(output * mask, target * mask) / (mask.sum() + 1e-8)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, mask, target in val_loader:
                x, mask, target = x.to(device), mask.to(device), target.to(device)
                output, _ = model(x, mask)
                
                # Compute loss only on missing values
                loss = F.mse_loss(output * mask, target * mask) / (mask.sum() + 1e-8)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_imputation_model.pt')
            
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return model

def evaluate_imputation(model, test_loader, device='cuda'):
    """Evaluate the imputation model"""
    
    model = model.to(device)
    model.eval()
    
    total_mse = 0
    total_mae = 0
    total_missing = 0
    
    with torch.no_grad():
        for x, mask, target in test_loader:
            x, mask, target = x.to(device), mask.to(device), target.to(device)
            output, _ = model(x, mask)
            
            # Compute metrics only on missing values
            mse = F.mse_loss(output * mask, target * mask, reduction='sum')
            mae = F.l1_loss(output * mask, target * mask, reduction='sum')
            
            total_mse += mse.item()
            total_mae += mae.item()
            total_missing += mask.sum().item()
    
    rmse = math.sqrt(total_mse / total_missing)
    mae = total_mae / total_missing
    
    print(f'Test RMSE: {rmse:.6f}, MAE: {mae:.6f}')
    
    return rmse, mae


################################################################################################################################


import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_physionet_data(file_path, missing_indicator=-1):
    """
    Load PhysioNet dataset and identify missing values
    
    Args:
        file_path: Path to the PhysioNet dataset
        missing_indicator: Value used to indicate missing data in the dataset
        
    Returns:
        data: DataFrame containing the data
        missing_mask: Binary mask where 1 indicates missing values
    """
    # Load data
    data = pd.read_csv(file_path)
    
    # Extract patient IDs and time steps if available
    id_cols = [col for col in data.columns if 'ID' in col.upper()]
    time_cols = [col for col in data.columns if 'TIME' in col.upper() or 'HOUR' in col.upper()]
    
    # Identify feature columns (non-ID, non-time columns)
    feature_cols = [col for col in data.columns if col not in id_cols + time_cols]
    
    # Create mask for missing values
    missing_mask = data[feature_cols].copy()
    if missing_indicator == -1:
        # If missing values are indicated by -1
        missing_mask = (missing_mask == missing_indicator).astype(float)
    else:
        # If missing values are indicated by NaN
        missing_mask = missing_mask.isna().astype(float)
    
    # Replace missing values with 0 for now (will be imputed by model)
    data_processed = data[feature_cols].copy()
    if missing_indicator == -1:
        data_processed = data_processed.replace(missing_indicator, 0)
    else:
        data_processed = data_processed.fillna(0)
    
    return data, data_processed, missing_mask, id_cols, time_cols, feature_cols

def preprocess_for_transformer(data, missing_mask, id_cols, time_cols, feature_cols, 
                              seq_len=48, normalize=True):
    """
    Preprocess data for transformer model
    
    Args:
        data: Original DataFrame containing all data
        data_processed: DataFrame with missing values replaced by 0
        missing_mask: DataFrame with 1 indicating missing values
        id_cols: List of ID column names
        time_cols: List of time column names
        feature_cols: List of feature column names
        seq_len: Sequence length for transformer
        normalize: Whether to normalize features
        
    Returns:
        X_tensor: Tensor of shape [num_patients, seq_len, num_features]
        mask_tensor: Missing mask tensor of same shape
        scaler: StandardScaler object (if normalize=True)
    """
    # If there are patient IDs, organize by patient
    if id_cols:
        patient_dfs = []
        mask_dfs = []
        
        for patient_id, patient_data in data.groupby(id_cols[0]):
            # Sort by time if time column exists
            if time_cols:
                patient_data = patient_data.sort_values(by=time_cols[0])
            
            # Extract features and mask
            patient_features = patient_data[feature_cols].copy()
            patient_mask = missing_mask.loc[patient_data.index].copy()
            
            # Pad or truncate to seq_len
            if len(patient_features) > seq_len:
                patient_features = patient_features.iloc[:seq_len]
                patient_mask = patient_mask.iloc[:seq_len]
            elif len(patient_features) < seq_len:
                # Pad with zeros (will be marked as missing)
                pad_length = seq_len - len(patient_features)
                padding = pd.DataFrame(0, index=range(pad_length), columns=feature_cols)
                patient_features = pd.concat([patient_features, padding])
                
                # Mark padding as missing in mask
                mask_padding = pd.DataFrame(0, index=range(pad_length), columns=feature_cols)
                patient_mask = pd.concat([patient_mask, mask_padding])
            
            patient_dfs.append(patient_features)
            mask_dfs.append(patient_mask)
        
        # Stack all patients
        data_array = np.stack([df.values for df in patient_dfs])
        mask_array = np.stack([df.values for df in mask_dfs])
    else:
        # If no patient IDs, reshape into fixed-length sequences
        values = data[feature_cols].values
        mask_values = missing_mask.values
        
        # Calculate number of complete sequences
        num_sequences = values.shape[0] // seq_len
        
        # Truncate to complete sequences
        values = values[:num_sequences * seq_len]
        mask_values = mask_values[:num_sequences * seq_len]
        
        # Reshape to [num_sequences, seq_len, num_features]
        data_array = values.reshape(num_sequences, seq_len, -1)
        mask_array = mask_values.reshape(num_sequences, seq_len, -1)
    
    # Normalize if requested
    if normalize:
        # Reshape for normalization
        original_shape = data_array.shape
        data_flat = data_array.reshape(-1, data_array.shape[-1])
        
        # Only use non-missing values for fitting scaler
        scaler = StandardScaler()
        # Create a mask for non-missing values
        non_missing_mask = ~(mask_array.reshape(-1, mask_array.shape[-1]).astype(bool))
        
        # For each feature, fit scaler on non-missing values
        for j in range(data_flat.shape[1]):
            non_missing_values = data_flat[non_missing_mask[:, j], j]
            if len(non_missing_values) > 0:  # Only fit if there are non-missing values
                feature_mean = non_missing_values.mean()
                feature_std = non_missing_values.std()
                
                # Avoid division by zero
                if feature_std == 0:
                    feature_std = 1
                
                # Manually normalize
                data_flat[:, j] = (data_flat[:, j] - feature_mean) / feature_std
        
        # Reshape back
        data_array = data_flat.reshape(original_shape)
    else:
        scaler = None
    
    # Convert to tensors
    X_tensor = torch.tensor(data_array, dtype=torch.float32)
    mask_tensor = torch.tensor(mask_array, dtype=torch.float32)
    
    return X_tensor, mask_tensor, scaler

def prepare_data_loaders(X, mask, batch_size=32, val_ratio=0.15, test_ratio=0.15):
    """
    Prepare DataLoaders for training, validation, and testing
    
    Args:
        X: Data tensor [num_samples, seq_len, feature_dim]
        mask: Missing mask tensor [num_samples, seq_len, feature_dim]
        batch_size: Batch size for training
        val_ratio: Ratio of data to use for validation
        test_ratio: Ratio of data to use for testing
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation, and testing
    """
    # Create target tensor (same as X but will be used as ground truth)
    target = X.clone()
    
    # Split data
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=val_ratio+test_ratio, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
    
    # Create datasets
    train_dataset = EHRDataset(X[train_idx], mask[train_idx])
    val_dataset = EHRDataset(X[val_idx], mask[val_idx])
    test_dataset = EHRDataset(X[test_idx], mask[test_idx])
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

# Example usage:
if __name__ == "__main__":
    file_path = "G:/FYP/FYP-EHR-Attention-Impute/FYP-Final/model/physionet_wo_missing.csv"
    
    # Load data
    data, data_processed, missing_mask, id_cols, time_cols, feature_cols = load_physionet_data(file_path)
    
    # Preprocess data
    X, mask, scaler = preprocess_for_transformer(
        data, missing_mask, id_cols, time_cols, feature_cols, seq_len=48
    )
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(X, mask, batch_size=32)
    
    # Initialize model
    feature_dim = X.shape[-1]
    model = EHRImputationTransformer(feature_dim, d_model=128, nhead=8, num_layers=4)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_imputation_model(model, train_loader, val_loader, epochs=50, device=device)
    
    # Evaluate model
    rmse, mae = evaluate_imputation(model, test_loader, device=device)
    
######################################################################################################################################################

import argparse
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import the model and data processing functions
from model import EHRImputationTransformer, train_imputation_model, evaluate_imputation

def create_artificial_missingness(data, mask, missing_percentage=0.2, missing_type='MCAR'):
    """
    Create artificial missingness for evaluation
    
    Args:
        data: Original data tensor
        mask: Original missing mask
        missing_percentage: Percentage of additional values to make missing
        missing_type: Type of missingness (MCAR, MAR, or MNAR)
        
    Returns:
        new_mask: Updated missing mask
    """
    new_mask = mask.clone()
    
    if missing_type == 'MCAR':
        # Missing Completely At Random
        artificial_mask = torch.bernoulli(torch.ones_like(data) * missing_percentage)
        # Don't make already missing values missing again
        artificial_mask = artificial_mask * (1 - mask)
        new_mask = new_mask + artificial_mask
        
    elif missing_type == 'MAR':
        # Missing At Random (depends on observed variables)
        # Here we make values more likely to be missing if nearby values are high
        nearby_values = torch.roll(data, shifts=1, dims=1)  # Shift by 1 in time dimension
        nearby_values[:, 0, :] = data[:, 0, :]  # Fix the first time step
        
        # Normalize to 0-1 range
        max_vals, _ = torch.max(torch.abs(nearby_values), dim=1, keepdim=True)
        max_vals = torch.clamp(max_vals, min=1e-8)  # Avoid division by zero
        nearby_values = torch.abs(nearby_values) / max_vals
        
        # Higher values = higher probability of being missing
        missing_probs = nearby_values * missing_percentage * 2  # Scale to get overall rate
        missing_probs = torch.clamp(missing_probs, max=0.8)  # Cap max probability
        
        artificial_mask = torch.bernoulli(missing_probs)
        # Don't make already missing values missing again
        artificial_mask = artificial_mask * (1 - mask)
        new_mask = new_mask + artificial_mask
        
    elif missing_type == 'MNAR':
        # Missing Not At Random (depends on the value itself)
        # Here we make higher values more likely to be missing
        
        # Normalize to 0-1 range
        max_vals, _ = torch.max(torch.abs(data), dim=1, keepdim=True)
        max_vals = torch.clamp(max_vals, min=1e-8)  # Avoid division by zero
        normalized_data = torch.abs(data) / max_vals
        
        # Higher values = higher probability of being missing
        missing_probs = normalized_data * missing_percentage * 2  # Scale to get overall rate
        missing_probs = torch.clamp(missing_probs, max=0.8)  # Cap max probability
        
        artificial_mask = torch.bernoulli(missing_probs)
        # Don't make already missing values missing again
        artificial_mask = artificial_mask * (1 - mask)
        new_mask = new_mask + artificial_mask
    
    return new_mask

def visualize_imputation(original_data, imputed_data, missing_mask, feature_names, time_steps=None, n_samples=3, n_features=5):
    """
    Visualize imputation results
    
    Args:
        original_data: Original data tensor [num_samples, seq_len, feature_dim]
        imputed_data: Imputed data tensor [num_samples, seq_len, feature_dim]
        missing_mask: Missing mask tensor [num_samples, seq_len, feature_dim]
        feature_names: List of feature names
        time_steps: List of time steps (optional)
        n_samples: Number of samples to visualize
        n_features: Number of features to visualize per sample
    """
    n_samples = min(n_samples, original_data.shape[0])
    n_features = min(n_features, original_data.shape[2])
    
    if time_steps is None:
        time_steps = list(range(original_data.shape[1]))
    
    # Sample patients and features
    sample_indices = np.random.choice(original_data.shape[0], n_samples, replace=False)
    feature_indices = np.random.choice(original_data.shape[2], n_features, replace=False)
    
    for i, sample_idx in enumerate(sample_indices):
        plt.figure(figsize=(15, n_features * 3))
        
        for j, feature_idx in enumerate(feature_indices):
            feature_name = feature_names[feature_idx] if feature_names else f"Feature {feature_idx}"
            
            # Get data for this sample and feature
            orig_values = original_data[sample_idx, :, feature_idx].cpu().numpy()
            imputed_values = imputed_data[sample_idx, :, feature_idx].cpu().numpy()
            mask_values = missing_mask[sample_idx, :, feature_idx].cpu().numpy()
            
            # Plot
            plt.subplot(n_features, 1, j + 1)
            
            # Plot original values
            plt.plot(time_steps, orig_values, 'b-', label='Original')
            
            # Plot imputed values for missing points
            missing_indices = np.where(mask_values > 0)[0]
            plt.plot(time_steps, imputed_values, 'r--', alpha=0.7, label='Imputed')
            plt.scatter([time_steps[idx] for idx in missing_indices], 
                      [imputed_values[idx] for idx in missing_indices], 
                      color='red', label='Imputed (missing)')
            
            plt.title(f"{feature_name}")
            plt.xlabel('Time step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'sample_{i+1}_imputation.png')
        plt.close()

def main(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{args.output_dir}/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    data, data_processed, missing_mask, id_cols, time_cols, feature_cols = load_physionet_data(
        args.data_path, missing_indicator=args.missing_indicator
    )
    
    # Save feature information
    feature_info = pd.DataFrame({
        'feature': feature_cols,
        'missing_rate': missing_mask.mean().values
    })
    feature_info.to_csv(f"{output_dir}/feature_info.csv", index=False)
    
    # Preprocess data
    print("Preprocessing data...")
    X, mask, scaler = preprocess_for_transformer(
        data, missing_mask, id_cols, time_cols, feature_cols, 
        seq_len=args.seq_len, normalize=not args.no_normalize
    )
    
    # Apply artificial missingness if specified
    if args.artificial_missing > 0:
        print(f"Adding artificial {args.missing_type} missingness ({args.artificial_missing:.1%})...")
        mask = create_artificial_missingness(
            X, mask, missing_percentage=args.artificial_missing, missing_type=args.missing_type
        )
    
    # Prepare data loaders
    print("Preparing data loaders...")
    train_loader, val_loader, test_loader = prepare_data_loaders(
        X, mask, batch_size=args.batch_size, 
        val_ratio=args.val_ratio, test_ratio=args.test_ratio
    )
    
    # Initialize model
    feature_dim = X.shape[-1]
    model = EHRImputationTransformer(
        feature_dim=feature_dim, 
        d_model=args.d_model, 
        nhead=args.nhead, 
        num_layers=args.num_layers, 
        dropout=args.dropout
    )
    
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {param_count:,} trainable parameters")
    
    # Train model
    print("Training model...")
    model = train_imputation_model(
        model, train_loader, val_loader, 
        epochs=args.epochs, lr=args.lr, device=device
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': vars(args),
        'feature_dim': feature_dim
    }, f"{output_dir}/model.pt")
    
    # Evaluate model
    print("Evaluating model...")
    rmse, mae = evaluate_imputation(model, test_loader, device=device)
    
    # Save evaluation results
    with open(f"{output_dir}/evaluation.txt", 'w') as f:
        f.write(f"RMSE: {rmse:.6f}\n")
        f.write(f"MAE: {mae:.6f}\n")
    
    # Visualize imputation on a subset of test data
    print("Visualizing imputation results...")
    with torch.no_grad():
        # Get a batch from test set
        batch = next(iter(test_loader))
        x, missing_mask, target = [b.to(device) for b in batch]
        
        # Get model predictions
        imputed, _ = model(x, missing_mask)
        
        # Visualize
        visualize_imputation(
            target, imputed, missing_mask, 
            feature_names=feature_cols, 
            n_samples=min(5, x.shape[0]), 
            n_features=min(5, x.shape[2])
        )
    
    print(f"Run completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EHR Imputation with Transformers')
    
    # Data arguments
    parser = argparse.ArgumentParser(description='EHR Imputation with Transformers')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True, help='Path to PhysioNet dataset')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--missing_indicator', type=float, default=float('nan'), 
                        help='Value indicating missing data (-1 or NaN)')
    parser.add_argument('--seq_len', type=int, default=48, help='Sequence length for transformer')
    parser.add_argument('--artificial_missing', type=float, default=0.0, 
                        help='Percentage of additional values to make missing (0.0-1.0)')
    parser.add_argument('--missing_type', type=str, default='MCAR', choices=['MCAR', 'MAR', 'MNAR'],
                        help='Type of artificial missingness')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=128, help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of transformer attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer encoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Other arguments
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--no_normalize', action='store_true', help='Disable feature normalization')
    
    args = parser.parse_args()
    main(args)
    
    
#####################################################################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

def evaluate_model_detailed(model, test_loader, device='cuda', output_dir=None):
    """
    Perform detailed evaluation of the imputation model
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        output_dir: Directory to save visualizations
        
    Returns:
        results_df: DataFrame with detailed results by feature
    """
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for x, mask, target in test_loader:
            x, mask, target = x.to(device), mask.to(device), target.to(device)
            output, _ = model(x, mask)
            
            # Store predictions, targets, and masks
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
    
    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Get total number of missing values
    total_missing = np.sum(all_masks)
    
    # Calculate overall metrics
    missing_preds = all_preds[all_masks > 0.5]
    missing_targets = all_targets[all_masks > 0.5]
    
    mse = mean_squared_error(missing_targets, missing_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(missing_targets, missing_preds)
    
    print(f"Overall RMSE: {rmse:.4f}")
    print(f"Overall MAE: {mae:.4f}")
    
    # Calculate metrics by feature
    n_features = all_preds.shape[2]
    feature_metrics = []
    
    for i in range(n_features):
        feature_mask = all_masks[:, :, i]
        feature_preds = all_preds[:, :, i][feature_mask > 0.5]
        feature_targets = all_targets[:, :, i][feature_mask > 0.5]
        
        if len(feature_preds) > 0:
            feature_mse = mean_squared_error(feature_targets, feature_preds)
            feature_rmse = np.sqrt(feature_mse)
            feature_mae = mean_absolute_error(feature_targets, feature_preds)
            feature_missing_count = np.sum(feature_mask)
            feature_missing_rate = np.mean(feature_mask)
        else:
            feature_rmse = np.nan
            feature_mae = np.nan
            feature_missing_count = 0
            feature_missing_rate = 0
        
        feature_metrics.append({
            'feature_index': i,
            'rmse': feature_rmse,
            'mae': feature_mae,
            'missing_count': feature_missing_count,
            'missing_rate': feature_missing_rate
        })
    
    # Create DataFrame with results
    results_df = pd.DataFrame(feature_metrics)
    
    # Plot feature-wise metrics
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot RMSE by feature
        plt.figure(figsize=(12, 6))
        sns.barplot(x='feature_index', y='rmse', data=results_df.sort_values('feature_index'))
        plt.title('RMSE by Feature')
        plt.xlabel('Feature Index')
        plt.ylabel('RMSE')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rmse_by_feature.png")
        plt.close()
        
        # Plot MAE by feature
        plt.figure(figsize=(12, 6))
        sns.barplot(x='feature_index', y='mae', data=results_df.sort_values('feature_index'))
        plt.title('MAE by Feature')
        plt.xlabel('Feature Index')
        plt.ylabel('MAE')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/mae_by_feature.png")
        plt.close()
        
        # Plot error distribution
        error = missing_preds - missing_targets
        plt.figure(figsize=(10, 6))
        sns.histplot(error, kde=True)
        plt.title('Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_distribution.png")
        plt.close()
        
        # Plot predicted vs actual
        plt.figure(figsize=(10, 6))
        plt.scatter(missing_targets, missing_preds, alpha=0.3)
        plt.plot([min(missing_targets), max(missing_targets)], 
                [min(missing_targets), max(missing_targets)], 'r--')
        plt.title('Predicted vs Actual')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pred_vs_actual.png")
        plt.close()
        
        # Save results to CSV
        results_df.to_csv(f"{output_dir}/feature_metrics.csv", index=False)
    
    return results_df

def analyze_embedding_space(model, test_loader, device='cuda', output_dir=None):
    """
    Analyze the embedding space of the transformer model
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run on
        output_dir: Directory to save visualizations
    """
    model = model.to(device)
    model.eval()
    
    all_embeddings = []
    all_masks = []
    
    # Extract embeddings from transformer
    with torch.no_grad():
        for x, mask, _ in test_loader:
            x, mask = x.to(device), mask.to(device)
            
            # Forward pass through embedding layer
            batch_size, seq_len, feature_dim = x.shape
            x_flat = x.reshape(batch_size, seq_len * feature_dim)
            mask_flat = mask.reshape(batch_size, seq_len * feature_dim)
            
            # Get embeddings (need to modify based on your model architecture)
            emb = model.embedding(x_flat.view(-1, feature_dim)).view(batch_size, seq_len, -1)
            emb = model.positional_encoding(emb)
            
            # Add mask embedding
            mask_emb = model.mask_embedding(mask_flat.long().view(-1, feature_dim).to(torch.int)).view(batch_size, seq_len, -1)
            emb = emb + mask_emb
            
            # Store embeddings and masks
            all_embeddings.append(emb.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
    
    # Concatenate results
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Reshape embeddings for visualization
    emb_shape = all_embeddings.shape
    all_embeddings = all_embeddings.reshape(-1, emb_shape[-1])
    all_masks = all_masks.reshape(-1, all_masks.shape[-1])
    
    # Calculate missing rate for each embedding
    missing_rates = np.mean(all_masks, axis=1)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(all_embeddings)
    
    # Apply t-SNE for better visualization
    tsne = TSNE(n_components=2, random_state=42)
    emb_tsne = tsne.fit_transform(all_embeddings[:5000])  # Limit to 5000 points for speed
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot PCA
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(emb_pca[:, 0], emb_pca[:, 1], c=missing_rates, 
                            cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Missing Rate')
        plt.title('PCA of Embedding Space')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/embedding_pca.png")
        plt.close()
        
        # Plot t-SNE
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(emb_tsne[:, 0], emb_tsne[:, 1], 
                            c=missing_rates[:5000], cmap='viridis', alpha=0.5)
        plt.colorbar(scatter, label='Missing Rate')
        plt.title('t-SNE of Embedding Space')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/embedding_tsne.png")
        plt.close()

def analyze_attention_weights(model, test_loader, device='cuda', output_dir=None, n_samples=5):
    """
    Analyze attention weights from the transformer model
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to run on
        output_dir: Directory to save visualizations
        n_samples: Number of samples to analyze
    """
    model = model.to(device)
    model.eval()
    
    # Register hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        attention_weights.append(output[1].detach().cpu())
    
    # Register hooks on all attention layers
    hooks = []
    for layer in model.transformer_encoder.layers:
        hooks.append(layer.self_attn.register_forward_hook(hook_fn))
    
    # Get a batch
    x, mask, _ = next(iter(test_loader))
    x, mask = x.to(device), mask.to(device)
    
    # Forward pass
    with torch.no_grad():
        model(x, mask)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize attention weights
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        for layer_idx, attn_weight in enumerate(attention_weights):
            for sample_idx in range(min(n_samples, attn_weight.shape[0])):
                weights = attn_weight[sample_idx].numpy()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(weights, cmap='viridis')
                plt.title(f'Attention Weights - Layer {layer_idx+1}, Sample {sample_idx+1}')
                plt.xlabel('Key Position')
                plt.ylabel('Query Position')
                plt.tight_layout()
                plt.savefig(f"{output_dir}/attn_layer{layer_idx+1}_sample{sample_idx+1}.png")
                plt.close()

def compare_imputation_methods(data, mask, transformer_model, device='cuda', output_dir=None):
    """
    Compare transformer imputation with simpler methods
    
    Args:
        data: Original data tensor [num_samples, seq_len, feature_dim]
        mask: Missing mask tensor [num_samples, seq_len, feature_dim]
        transformer_model: Trained transformer model
        device: Device to run on
        output_dir: Directory to save visualizations
    """
    X_np = data.numpy()
    mask_np = mask.numpy()
    
    # Apply mean imputation
    mean_imputed = np.copy(X_np)
    for j in range(X_np.shape[2]):
        feature_mean = np.nanmean(X_np[:, :, j][mask_np[:, :, j] == 0])
        missing_indices = mask_np[:, :, j] == 1
        mean_imputed[:, :, j][missing_indices] = feature_mean
    
    # Apply forward fill imputation
    ffill_imputed = np.copy(X_np)
    for i in range(X_np.shape[0]):
        for j in range(X_np.shape[2]):
            values = X_np[i, :, j]
            mask_values = mask_np[i, :, j]
            last_valid = np.nan
            
            for t in range(len(values)):
                if mask_values[t] == 0:
                    last_valid = values[t]
                elif not np.isnan(last_valid):
                    ffill_imputed[i, t, j] = last_valid
    
    # Apply transformer imputation
    transformer_imputed = np.copy(X_np)
    transformer_model = transformer_model.to(device)
    transformer_model.eval()
    
    with torch.no_grad():
        x_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).to(device)
        
        # Impute in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(X_np), batch_size):
            end_idx = min(i + batch_size, len(X_np))
            batch_x = x_tensor[i:end_idx]
            batch_mask = mask_tensor[i:end_idx]
            
            output, _ = transformer_model(batch_x, batch_mask)
            transformer_imputed[i:end_idx] = output.cpu().numpy()
    
    # Calculate metrics for each method
    # Use only missing values for evaluation
    missing_indices = mask_np == 1
    true_values = X_np[missing_indices]
    
    mean_values = mean_imputed[missing_indices]
    ffill_values = ffill_imputed[missing_indices]
    transformer_values = transformer_imputed[missing_indices]
    
    mean_rmse = np.sqrt(mean_squared_error(true_values, mean_values))
    ffill_rmse = np.sqrt(mean_squared_error(true_values, ffill_values))
    transformer_rmse = np.sqrt(mean_squared_error(true_values, transformer_values))
    
    mean_mae = mean_absolute_error(true_values, mean_values)
    ffill_mae = mean_absolute_error(true_values, ffill_values)
    transformer_mae = mean_absolute_error(true_values, transformer_values)
    
    print(f"Mean Imputation - RMSE: {mean_rmse:.4f}, MAE: {mean_mae:.4f}")
    print(f"Forward Fill - RMSE: {ffill_rmse:.4f}, MAE: {ffill_mae:.4f}")
    print(f"Transformer - RMSE: {transformer_rmse:.4f}, MAE: {transformer_mae:.4f}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame for comparison
        methods = ['Mean Imputation', 'Forward Fill', 'Transformer']
        rmse_values = [mean_rmse, ffill_rmse, transformer_rmse]
        mae_values = [mean_mae, ffill_mae, transformer_mae]
        
        comparison_df = pd.DataFrame({
            'Method': methods,
            'RMSE': rmse_values,
            'MAE': mae_values
        })
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        
        bar_width = 0.35
        index = np.arange(len(methods))
        
        plt.bar(index, rmse_values, bar_width, label='RMSE')
        plt.bar(index + bar_width, mae_values, bar_width, label='MAE')
        
        plt.xlabel('Imputation Method')
        plt.ylabel('Error')
        plt.title('Comparison of Imputation Methods')
        plt.xticks(index + bar_width / 2, methods)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/method_comparison.png")
        plt.close()
        
        # Save comparison to CSV
        comparison_df.to_csv(f"{output_dir}/method_comparison.csv", index=False)
        
        # Visualize examples of imputation
        # Select a sample with missing values
        sample_idx = np.where(np.sum(mask_np, axis=(1, 2)) > 0)[0][0]
        
        sample_data = X_np[sample_idx]
        sample_mask = mask_np[sample_idx]
        sample_mean = mean_imputed[sample_idx]
        sample_ffill = ffill_imputed[sample_idx]
        sample_transformer = transformer_imputed[sample_idx]
        
        # Select a feature with missing values
        feature_idx = np.where(np.sum(sample_mask, axis=0) > 0)[0][0]
        
        # Plot the feature over time
        plt.figure(figsize=(12, 6))
        
        # Original data (only observed points)
        observed_indices = sample_mask[:, feature_idx] == 0
        plt.scatter(np.where(observed_indices)[0], sample_data[observed_indices, feature_idx], 
                  color='black', label='Observed', s=50, zorder=5)
        
        # True values for missing points
        missing_indices = sample_mask[:, feature_idx] == 1
        plt.scatter(np.where(missing_indices)[0], sample_data[missing_indices, feature_idx],
                  color='black', marker='X', label='True (Missing)', s=100, zorder=4)
        
        # Mean imputation
        plt.scatter(np.where(missing_indices)[0], sample_mean[missing_indices, feature_idx],
                  color='blue', marker='o', label='Mean', s=50, zorder=3)
        
        # Forward fill imputation
        plt.scatter(np.where(missing_indices)[0], sample_ffill[missing_indices, feature_idx],
                  color='green', marker='o', label='Forward Fill', s=50, zorder=2)
        
        # Transformer imputation
        plt.scatter(np.where(missing_indices)[0], sample_transformer[missing_indices, feature_idx],
                  color='red', marker='o', label='Transformer', s=50, zorder=1)
        
        plt.title(f'Comparison of Imputation Methods - Feature {feature_idx}')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/imputation_example.png")
        plt.close()
        
    return {
        'mean_rmse': mean_rmse,
        'ffill_rmse': ffill_rmse,
        'transformer_rmse': transformer_rmse,
        'mean_mae': mean_mae,
        'ffill_mae': ffill_mae,
        'transformer_mae': transformer_mae
    }