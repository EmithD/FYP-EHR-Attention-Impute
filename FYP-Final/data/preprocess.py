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
    file_path = "path_to_physionet_dataset.csv"
    
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