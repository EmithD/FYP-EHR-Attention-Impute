import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EHRDataPreprocessor:
    def __init__(self):
        self.numeric_columns = [
            'temperature', 'heart_rate', 'blood_pressure_systolic',
            'blood_pressure_diastolic', 'respiratory_rate', 'oxygen_saturation',
            'wbc_count', 'rbc_count', 'hemoglobin', 'hematocrit',
            'platelet_count', 'sodium', 'potassium', 'chloride', 'glucose'
        ]
        self.categorical_columns = ['gender', 'code']
        self.scalers = {}
        
    def fit_transform(self, df):
        """Standardize numeric features and encode categorical features"""
        processed_data = {}
        
        # Standardize numeric columns
        for col in self.numeric_columns:
            scaler = StandardScaler()
            # Fit scaler only on non-missing values
            valid_data = df[col].dropna()
            scaler.fit(valid_data.values.reshape(-1, 1))
            self.scalers[col] = scaler
            
            # Transform the entire column, leaving NaN values as is
            transformed_values = df[col].values.reshape(-1, 1)
            mask = ~np.isnan(transformed_values)
            transformed_values[mask] = scaler.transform(transformed_values[mask])
            processed_data[col] = transformed_values.flatten()
        
        # Encode categorical columns
        processed_data['gender'] = (df['gender'] == 'M').astype(float)  # Binary encoding
        # One-hot encode diagnosis codes
        diagnosis_dummies = pd.get_dummies(df['code'], prefix='diagnosis')
        for col in diagnosis_dummies.columns:
            processed_data[col] = diagnosis_dummies[col].values
            
        # Convert to tensor
        features = []
        for col in processed_data:
            features.append(torch.FloatTensor(processed_data[col]).unsqueeze(1))
        
        return torch.cat(features, dim=1).to(device)
    
    def inverse_transform(self, tensor_data):
        """Convert standardized values back to original scale"""
        transformed_data = {}
        current_idx = 0
        
        # Convert numeric columns back to original scale
        for col in self.numeric_columns:
            transformed_data[col] = self.scalers[col].inverse_transform(
                tensor_data[:, current_idx].cpu().numpy().reshape(-1, 1)
            ).flatten()
            current_idx += 1
            
        return transformed_data

class EHRDataset(Dataset):
    def __init__(self, features, mask):
        self.features = features  # Standardized features
        self.mask = mask  # Binary mask indicating missing values
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.mask[idx]
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=5000):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0)]