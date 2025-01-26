import torch
import torch.nn as nn

# class TransformerImputer(nn.Module):
#     def __init__(self, input_dim, embed_dim, num_heads, num_layers, dropout=0.1):
#         super(TransformerImputer, self).__init__()

#         # Embedding layers
#         self.input_embedding = nn.Linear(input_dim, embed_dim)
#         self.position_embedding = nn.Embedding(500, embed_dim)  # Max sequence length set to 500

#         # Learnable mask token
#         self.mask_token = nn.Parameter(torch.randn(embed_dim))

#         # Transformer encoder layers
#         self.transformer_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(
#                 d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=dropout, activation="gelu"
#             ) for _ in range(num_layers)
#         ])
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer=self.transformer_layers[0], num_layers=num_layers
#         )

#         # Fully connected layers for imputation
#         self.fc = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim * 4, embed_dim * 2),
#             nn.GELU(),
#             nn.Linear(embed_dim * 2, input_dim),
#         )

#         # Layer normalization
#         self.layer_norm = nn.LayerNorm(embed_dim)

#         # Auxiliary loss layers for masked language modeling
#         self.mlm_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 2),
#             nn.GELU(),
#             nn.Linear(embed_dim * 2, input_dim)
#         )

#         # Attention weights storage for interpretability
#         self.attention_weights = []

#     def forward(self, x, mask):
#         """
#         Args:
#             x: Input tensor of shape (batch_size, seq_len, input_dim)
#             mask: Binary mask of shape (batch_size, seq_len) with 1 for missing values and 0 otherwise
#         Returns:
#             Imputed data tensor of shape (batch_size, seq_len, input_dim)
#         """
#         batch_size, seq_len, _ = x.shape

#         # Replace missing values with the mask token
#         x[mask == 1] = 0.0
#         x_embed = self.input_embedding(x)  # Embed input data

#         # Add positional encodings
#         positions = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
#         pos_embed = self.position_embedding(positions)

#         x_embed = self.layer_norm(x_embed + pos_embed)

#         # Pass through transformer encoder and collect attention weights
#         self.attention_weights = []
#         for layer in self.transformer_layers:
#             x_embed = layer(x_embed.permute(1, 0, 2))  # (seq_len, batch_size, embed_dim)
#             self.attention_weights.append(layer.self_attn.attn_output_weights.detach().cpu())
#         x_transformed = x_embed.permute(1, 0, 2)  # Back to (batch_size, seq_len, embed_dim)

#         # Decode to original input dimension
#         x_imputed = self.fc(x_transformed)

#         # Replace missing values with model predictions
#         x[mask == 1] = x_imputed[mask == 1]

#         # Auxiliary loss output (optional)
#         mlm_logits = self.mlm_head(x_transformed)

#         return x, mlm_logits


# class TransformerImputer(nn.Module):
#     def __init__(self, input_dim, num_heads=4, hidden_dim=64, num_layers=2):
#         super(TransformerImputer, self).__init__()
#         self.input_embedding = nn.Linear(input_dim, hidden_dim)
#         self.positional_encoding = nn.Parameter(torch.zeros(1, input_dim, hidden_dim))
        
#         self.transformer = nn.Transformer(
#             d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers
#         )

#         self.output_layer = nn.Linear(hidden_dim, input_dim)

#     def forward(self, x):
#         embedded_input = self.input_embedding(x) + self.positional_encoding
#         transformer_output = self.transformer(embedded_input, embedded_input)
#         output = self.output_layer(transformer_output)
#         return output


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Step 1: Pre-process Data
class TabularDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def preprocess_data(dataset):
    # Normalize the dataset
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(dataset)

    # Train-test split
    train_data, test_data = train_test_split(normalized_data, test_size=0.2, random_state=42)
    return train_data, test_data, scaler

# Step 2: Create Transformer Model
class TransformerImputer(nn.Module):
    def __init__(self, input_dim, num_heads=4, hidden_dim=64, num_layers=2):
        super(TransformerImputer, self).__init__()
        self.input_embedding = nn.Linear(input_dim, hidden_dim)
        
        self.transformer = nn.Transformer(
            d_model=hidden_dim, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers
        )

        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # Add positional encoding dynamically based on input size
        batch_size, seq_len = x.size()
        positional_encoding = torch.zeros(batch_size, seq_len, self.input_embedding.out_features, device=x.device)

        embedded_input = self.input_embedding(x) + positional_encoding
        transformer_output = self.transformer(embedded_input.transpose(0, 1), embedded_input.transpose(0, 1))
        output = self.output_layer(transformer_output.transpose(0, 1))
        return output

# Step 3: Train the Model
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# Step 4: Test with Artificial Missingness
def add_artificial_missingness(data, missing_rate=0.1):
    mask = np.random.binomial(1, 1 - missing_rate, data.shape)
    masked_data = data * mask
    return masked_data, mask

# Step 5: Evaluate Results
def evaluate_model(model, test_data, mask, scaler):
    model.eval()
    with torch.no_grad():
        imputed_data = model(torch.tensor(test_data, dtype=torch.float32)).numpy()
        imputed_data = scaler.inverse_transform(imputed_data)
        original_data = scaler.inverse_transform(test_data)

    rmse = np.sqrt(mean_squared_error(original_data[mask == 0], imputed_data[mask == 0]))
    print(f"RMSE: {rmse:.4f}")

# Load Existing Dataset
def load_dataset(file_path):
    dataset = pd.read_csv(file_path)
    return dataset.values

# Example Usage
data = load_dataset("../data/physionet_wo_missing.csv")  # Replace with your dataset file path
train_data, test_data, scaler = preprocess_data(data)

train_dataset = TabularDataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

model = TransformerImputer(input_dim=data.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, criterion, optimizer, epochs=20)

# Adding artificial missingness to test data
masked_test_data, mask = add_artificial_missingness(test_data)

# Evaluate model
evaluate_model(model, masked_test_data, mask, scaler)