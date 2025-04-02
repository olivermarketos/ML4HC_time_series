import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import RobustScaler, StandardScaler
import random
import matplotlib.pyplot as plt
import time

MAX_SEQ_LEN = 768
PAD_VALUE = -999.0    # Padding value for time and value features
PAD_INDEX_Z = 0       # Padding index for modality (ensure 0 is reserved)


class MedicalTimeSeriesDatasetTimeGrid(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx].unsqueeze(-1)
    
class MedicalTimeSeriesDatasetTuple(Dataset):
    def __init__(self, data_triplets, labels, max_seq_len=MAX_SEQ_LEN):
        self.data_triplets = data_triplets # List of lists of triplets
        self.labels = labels               # List of labels
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        triplets = self.data_triplets[idx]
        label = self.labels[idx]

        if len(triplets) > self.max_seq_len:
            triplets = triplets[:self.max_seq_len]
        # Separate into t, z, v sequences
        t_seq = torch.tensor([t[0] for t in triplets], dtype=torch.float32)
        z_seq = torch.tensor([t[1] for t in triplets], dtype=torch.long)
        v_seq = torch.tensor([t[2] for t in triplets], dtype=torch.float32)

        return t_seq, z_seq, v_seq, torch.tensor(label, dtype=torch.float32)

def collate_fn(batch):
    # Unzip the batch
    t_seqs, z_seqs, v_seqs, labels = zip(*batch)

    # Pad sequences to the maximum length in the batch
    t_seqs_padded = nn.utils.rnn.pad_sequence(t_seqs, batch_first=True, padding_value=PAD_VALUE)
    z_seqs_padded = nn.utils.rnn.pad_sequence(z_seqs, batch_first=True, padding_value=PAD_INDEX_Z)
    v_seqs_padded = nn.utils.rnn.pad_sequence(v_seqs, batch_first=True, padding_value=PAD_VALUE)


    # Calculate the mask directly in the format expected by TransformerEncoder: True = IGNORE
    # A position is ignored IF it's a padding index  
    attn_mask = (z_seqs_padded == PAD_INDEX_Z)

    labels = torch.stack(labels)

    return t_seqs_padded, z_seqs_padded, v_seqs_padded, attn_mask, labels


class PositionalEncodingTimeGrid(nn.Module):

    def __init__(self, d_model,dropout = 0.1,  max_len=49):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) # max_length: number tokens, d_model: dimension of each token (embedding dim)

        position = torch.arange(0, max_len).unsqueeze(1) # shape (max_length, 1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)* -(torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))
        self.pe: torch.Tensor

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return self.dropout(x)

# class PositionalEncodingTuple(nn.Module):
#     """Standard sinusoidal positional encoding."""
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]

#         return self.dropout(x)


class TimeSeriesGridTransformer(nn.Module):

    def __init__(self, feature_dim = 41, d_model = 128, nhead = 4, num_layers = 2,dropout=0.1) -> None:
        super().__init__()

        self.d_model = d_model
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncodingTimeGrid(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)


        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
            )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
      
        x = x[:,-1, :] # last time step

        x = self.classifier(x)
        return x
    

class TimeSeriesTupleTransformer(nn.Module):
    def __init__(self, num_modalities, d_model, nhead, num_encoder_layers,
                 dim_feedforward, num_classes, dropout=0.1, max_seq_len=MAX_SEQ_LEN, modality_emb_dim=64):
        super().__init__()
        self.d_model = d_model
        self.modality_emb_dim = modality_emb_dim


        # Input embeddings
        self.modality_embedding = nn.Embedding(num_modalities, self.modality_emb_dim, padding_idx=PAD_INDEX_Z)

        combined_input_dim = 1 + 1 + self.modality_emb_dim 
        self.input_proj = nn.Linear(combined_input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Classifier head
        self.classifier = nn.Linear(d_model, num_classes)

    #     self.init_weights()

    # def init_weights(self):
    #     initrange = 0.1
    #     self.modality_embedding.weight.data.uniform_(-initrange, initrange)
    #     self.time_encoder_linear.weight.data.uniform_(-initrange, initrange)
    #     self.value_encoder_linear.weight.data.uniform_(-initrange, initrange)
    #     self.modality_adjust_linear.weight.data.uniform_(-initrange, initrange)
    #     # self.input_proj.weight.data.uniform_(-initrange, initrange)
    #     # self.input_proj.bias.data.zero_()
    #     self.classifier.weight.data.uniform_(-initrange, initrange)
    #     self.classifier.bias.data.zero_()

    def forward(self, t_seq, z_seq, v_seq, src_key_padding_mask):
        """
        Args:
            t_seq: (batch_size, seq_len) - Scaled time
            z_seq: (batch_size, seq_len) - Modality indices
            v_seq: (batch_size, seq_len) - Scaled values
            src_key_padding_mask: (batch_size, seq_len) - Bool mask (True=ignore)
        """
        # Embed inputs
        mod_embed = self.modality_embedding(z_seq) 

        t_feat = t_seq.unsqueeze(-1) # (B, S, 1)
        v_feat = v_seq.unsqueeze(-1)

        combined_inputs = torch.cat([t_feat, v_feat, mod_embed], dim=-1) # (B, S, D)

        projected_emb = self.input_proj(combined_inputs) # (B, S, D)

        final_emb = projected_emb # renamed for clarity

        # Transformer Encoder
        # src_key_padding_mask should be True for positions to ignore
        transformer_output = self.transformer_encoder(final_emb, 
                                                    src_key_padding_mask=src_key_padding_mask) # true = ignore

        
        mask_float = (~src_key_padding_mask).unsqueeze(-1).float() # (B, S, 1), True=valid -> 1.0, False=ignore -> 0.0
        masked_sum = (transformer_output * mask_float).sum(dim=1) # (B, D)
        num_valid = mask_float.sum(dim=1).clamp(min=1e-9) # (B, 1)
        pooled_output = masked_sum / num_valid # (B, D)

        # Classification
        logits = self.classifier(pooled_output) # (B, num_classes)

        return logits



