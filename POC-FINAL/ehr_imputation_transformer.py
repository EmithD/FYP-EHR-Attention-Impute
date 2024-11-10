import torch.nn as nn
from positional_encoding import PositionalEncoding

class EHRImputationTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(EHRImputationTransformer, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True),
            num_layers=num_encoder_layers
        )
        self.output_layer = nn.Linear(d_model, input_dim)

    def forward(self, x):
        
        assert x.size(2) == self.input_embedding.in_features, "Input feature size mismatch"
        
        x = self.input_embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.output_layer(x)
        return x
