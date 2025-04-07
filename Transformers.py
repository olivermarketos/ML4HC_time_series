import numpy as np
import pandas as pd
from numpy.random import normal
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
    def __init__(self, data_triplets, labels, max_seq_len=578):
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

class ContrastiveMedicalTimeSeriesGridDataset(Dataset):
    def __init__(self, X, augmentation_func):
        """
        Args:
            X (numpy.ndarray or torch.Tensor): The input data array of shape (num_samples, seq_len, features).
            augmentation_func (callable): A function that takes a tensor of shape (seq_len, features)
                                          and returns an augmented tensor of the same shape.
        """
        # Ensure X is a tensor
        if isinstance(X, np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float32)
        else:
            self.X = X.float() # Assuming X might already be a tensor

        self.augmentation_func = augmentation_func
        
        # --- Pre-computation Check (Optional but Recommended) ---
        # Check if dimensions match expectations (e.g., seq_len=49, features=41)
        if len(self.X.shape) != 3:
            raise ValueError(f"Input data X must have 3 dimensions (samples, seq_len, features), but got shape {self.X.shape}")
        print(f"Initialized Contrastive Grid Dataset with data shape: {self.X.shape}")
        # Example check: if self.X.shape[1] != 49 or self.X.shape[2] != 41:
        #    warnings.warn("Input data shape doesn't match expected (49, 41)")


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        original_sample = self.X[idx] # Shape: (seq_len, features) e.g., (49, 41)

        # Create two independent augmentations (views)
        view1 = self.augmentation_func(original_sample)
        view2 = self.augmentation_func(original_sample)

        return view1, view2

class ContrastiveMedicalTimeSeriesDatasetTuple(Dataset):

    def __init__(self, data_triplets, labels, max_seq_len=578, augmentation=None):
        self.data_triplets = data_triplets # List of lists of triplets
        self.labels = labels               # List of labels
        self.max_seq_len = max_seq_len
        self.augmentation = augmentation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        triplets = self.data_triplets[idx]
        label = self.labels[idx]

        if len(triplets) > self.max_seq_len:
            triplets = triplets[:self.max_seq_len]
       
        view1 = self.augmentation(triplets) if self.augmentation else triplets
        view2 = self.augmentation(triplets) if self.augmentation else triplets

        def process(triplets_view):
            t_seq = torch.tensor([t[0] for t in triplets_view], dtype=torch.float32)
            z_seq = torch.tensor([t[1] for t in triplets_view], dtype=torch.long)
            v_seq = torch.tensor([t[2] for t in triplets_view], dtype=torch.float32)
            return t_seq, z_seq, v_seq

        t_seq1, z_seq1, v_seq1 = process(view1)
        t_seq2, z_seq2, v_seq2 = process(view2)

        return (t_seq1, z_seq1, v_seq1), (t_seq2, z_seq2, v_seq2), torch.tensor(label, dtype=torch.float32)       

def gaussian_noise(x, std=0.05):
    """Adds Gaussian noise to the tensor."""
    noise = torch.randn_like(x) * std
    return x + noise

def time_masking(x, max_mask_len_ratio=0.2, mask_value=0.0):
    """Randomly masks a contiguous block of time steps."""
    seq_len = x.shape[0]
    mask_len = random.randint(1, int(seq_len * max_mask_len_ratio))
    mask_start = random.randint(0, seq_len - mask_len)
    
    x_aug = x.clone()
    x_aug[mask_start : mask_start + mask_len, :] = mask_value
    return x_aug

def feature_masking(x, max_mask_len_ratio=0.2, mask_value=0.0):
    """Randomly masks entire features (columns)."""
    num_features = x.shape[1]
    mask_len = random.randint(1, int(num_features * max_mask_len_ratio))
    # Choose features (columns) to mask
    masked_features = random.sample(range(num_features), mask_len)
    
    x_aug = x.clone()
    x_aug[:, masked_features] = mask_value
    return x_aug

def random_permutation(x, segment_len=5):
    """Randomly permutes segments along the time axis."""
    seq_len = x.shape[0]
    num_segments = seq_len // segment_len
    if num_segments < 2:
        return x # Not enough segments to permute

    indices = list(range(num_segments))
    random.shuffle(indices)
    
    x_aug = x.clone()
    original_segments = [x[i*segment_len : (i+1)*segment_len, :] for i in range(num_segments)]
    
    # Handle leftover part if seq_len is not divisible by segment_len
    leftover_len = seq_len % segment_len
    leftover_segment = x[num_segments*segment_len:, :] if leftover_len > 0 else None

    # Reconstruct based on shuffled indices
    current_pos = 0
    for i in indices:
        segment = original_segments[i]
        x_aug[current_pos : current_pos + segment_len, :] = segment
        current_pos += segment_len
        
    # Add back leftover part
    if leftover_segment is not None:
         x_aug[current_pos:, :] = leftover_segment
         
    return x_aug

# --- Composite Augmentation Function ---
# You can create a function that randomly applies one or more augmentations

def apply_grid_augmentations(x):
    """Applies a randomly chosen augmentation or sequence."""
    # Option 1: Choose one augmentation randomly
    # aug_choice = random.choice(['noise', 'time_mask', 'feature_mask', 'permute'])
    # if aug_choice == 'noise':
    #     return gaussian_noise(x)
    # elif aug_choice == 'time_mask':
    #     return time_masking(x)
    # elif aug_choice == 'feature_mask':
    #     return feature_masking(x)
    # elif aug_choice == 'permute':
    #     return random_permutation(x)

    # Option 2: Apply a sequence (e.g., noise then time masking)
    x_aug = gaussian_noise(x, std=0.02) # Apply mild noise first
    x_aug = time_masking(x_aug, max_mask_len_ratio=0.15) # Then mask time steps
    # x_aug = feature_masking(x_aug, max_mask_len_ratio=0.1) # Optionally mask features
    
    return x_aug


def collate_fn(batch):
    # Unzip the batch
    t_seqs, z_seqs, v_seqs, labels_list = zip(*batch)

    # Pad sequences to the maximum length in the batch
    t_seqs_padded = nn.utils.rnn.pad_sequence(list(t_seqs), batch_first=True, padding_value=PAD_VALUE)
    z_seqs_padded = nn.utils.rnn.pad_sequence(list(z_seqs), batch_first=True, padding_value=PAD_INDEX_Z)
    v_seqs_padded = nn.utils.rnn.pad_sequence(list(v_seqs), batch_first=True, padding_value=PAD_VALUE)


    # Calculate the mask directly in the format expected by TransformerEncoder: True = IGNORE
    # A position is ignored IF it's a padding index  
    attn_mask = (z_seqs_padded == PAD_INDEX_Z)

    labels = torch.stack(labels_list) # Stack labels to form a tensor
    
    if labels.ndim == 1: # Check if it's indeed [B]
        labels = labels.unsqueeze(-1)

    # Ensure float type for loss calculation later
    labels = labels.float()
    return t_seqs_padded, z_seqs_padded, v_seqs_padded, attn_mask, labels

def augment_triplet(triplets, time_shift_max = 0.05, jitter_std = 0.01, dropout_prob = 0.1):

    augmented = []
    for t, z, v in triplets:
        # time shift
        t_aug = t+ random.uniform(-time_shift_max, time_shift_max)

        # jitter value
        v_aug = v + np.random.normal(0, jitter_std) # jitter

        if random.random() < dropout_prob:
            v_aug = 0.0

        augmented.append((t_aug, z, v_aug))
    return augmented


def contrastive_grid_collate_fn(batch):
    """
    Collates batches for contrastive learning with grid data (fixed dimensions).
    Args:
        batch: A list of tuples, where each tuple is (view1, view2).
               view1 and view2 are tensors of shape (seq_len, features).
    Returns:
        A tuple (views1_batch, views2_batch), where each element is a tensor
        of shape (batch_size, seq_len, features).
    """
    view1_list, view2_list = zip(*batch)

    # Stack the tensors along the batch dimension
    views1_batch = torch.stack(view1_list, dim=0)
    views2_batch = torch.stack(view2_list, dim=0)

    return views1_batch, views2_batch
       
def contrastive_collate_fn(batch):
    view1, view2, labels_list = zip(*batch)
    # Unzip the views
    t_seqs1, z_seqs1, v_seqs1 = zip(*view1)
    t_seqs2, z_seqs2, v_seqs2 = zip(*view2)

    # Pad sequences to the maximum length in the batch
    t_seqs1_padded = nn.utils.rnn.pad_sequence(list(t_seqs1), batch_first=True, padding_value=PAD_VALUE)
    v_seqs1_padded = nn.utils.rnn.pad_sequence(list(v_seqs1), batch_first=True, padding_value=PAD_VALUE)
    z_seqs1_padded = nn.utils.rnn.pad_sequence(list(z_seqs1), batch_first=True, padding_value=PAD_INDEX_Z)
    attn_mask1 = (z_seqs1_padded == PAD_INDEX_Z)

    t_seqs2_padded = nn.utils.rnn.pad_sequence(list(t_seqs2), batch_first=True, padding_value=PAD_VALUE)
    v_seqs2_padded = nn.utils.rnn.pad_sequence(list(v_seqs2), batch_first=True, padding_value=PAD_VALUE)
    z_seqs2_padded = nn.utils.rnn.pad_sequence(list(z_seqs2), batch_first=True, padding_value=PAD_INDEX_Z)
    attn_mask2 = (z_seqs2_padded == PAD_INDEX_Z)

    labels = torch.stack(labels_list) # Stack labels to form a tensor
    if labels.ndim == 1: # Check if it's indeed [B]
        labels = labels.unsqueeze(-1)
    labels = labels.float()

    return (t_seqs1_padded, z_seqs1_padded, v_seqs1_padded, attn_mask1), (t_seqs2_padded, z_seqs2_padded, v_seqs2_padded, attn_mask2), labels

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

class ContinuousTimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, t):
        # t: (batch_size, seq_len) with values scaled between 0 and 1
        device = t.device
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device) * (-np.log(10000.0) / self.d_model)
        )
        t_unsqueezed = t.unsqueeze(-1)  # shape: (B, S, 1)
        pe = torch.zeros(t.shape[0], t.shape[1], self.d_model, device=device)
        pe[..., 0::2] = torch.sin(t_unsqueezed * div_term)
        pe[..., 1::2] = torch.cos(t_unsqueezed * div_term)
        return pe
    
class PositionalEncodingTuple(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.pe: torch.Tensor

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]

        return self.dropout(x)


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
        
    def get_representation(self, x):
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)      
        x = x[:,-1, :] # last time step
        return x

    def forward(self, x):
        rep = self.get_representation(x)
       

        x = self.classifier(rep)
        return x
    
class TimeSeriesTupleTransformer(nn.Module):
    def __init__(self, num_modalities, d_model, nhead, num_encoder_layers,
                 dim_feedforward, num_classes, dropout=0.1, max_seq_len=768, modality_emb_dim=64):
        super().__init__()
        self.d_model = d_model

        self.modality_emb_dim = modality_emb_dim


        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))  # Class token for classification
        self.modality_embedding = nn.Embedding(num_modalities, self.modality_emb_dim, padding_idx=PAD_INDEX_Z)
        self.feature_proj = nn.Linear(self.modality_emb_dim, self.d_model)  # 1 for time, modality_emb_dim for modality embedding

        self.time_encoding = ContinuousTimeEncoding(d_model)
        self.pos_encoder = PositionalEncodingTuple(d_model, dropout, max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
            )

    def get_representation(self, t_seq, z_seq, v_seq, src_key_padding_mask):

        # NEW REPRESENTATION
        mod_embed = self.modality_embedding(z_seq)  # (B, S, modality_emb_dim)
        v_unsqueezed = v_seq.unsqueeze(-1)                # (B, S, 1)
        x = mod_embed * v_unsqueezed  # (B, S, 1+modality_emb_dim)
        x = self.feature_proj(x)  # (B, S, d_model)
        time_emb = self.time_encoding(t_seq)  # (B, S, d_model)
        x = x + time_emb  # (B, S, d_model)
        x = self.pos_encoder(x)  # (B, S, d_model)

        output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, S, d_model)
        attn_weights = torch.softmax(torch.mean(output, dim=-1), dim=1).unsqueeze(-1)  # (B, S, 1)
        pooled = torch.sum(output * attn_weights, dim=1)  # (B, d_model)
        return pooled

        # ORIGINAL REPRESENTATION
        # mod_embed = self.modality_embedding(z_seq)  # (B, S, modality_emb_dim)
        # t_feat = t_seq.unsqueeze(-1)                # (B, S, 1)
        # v_feat = v_seq.unsqueeze(-1)                # (B, S, 1)
        # combined_inputs = torch.cat([t_feat, v_feat, mod_embed], dim=-1)  # (B, S, 1+1+modality_emb_dim)
        # projected_emb = self.input_proj(combined_inputs)                  # (B, S, d_model)
       
        # time_emb = self.time_encoding(t_seq)  # (B, S, d_model)

        # x = projected_emb + time_emb  # (B, S, d_model)

        # batch_size = projected_emb.size(0)
        # cls_token = self.cls_token.expand(batch_size, -1, -1)             # (B, 1, d_model)
        # x = torch.cat((cls_token, projected_emb), dim=1)                  # (B, S+1, d_model)

        # cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=src_key_padding_mask.device)
        # src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)  # (B, 1+S)

        # transformer_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, 1+S, d_model)
        # cls_representation = transformer_output[:, 0, :]  # (B, d_model), output corresponding to CLS token
        # return cls_representation


        final_emb = projected_emb # renamed for clarity
        transformer_output = self.transformer_encoder(final_emb, 
                                                    src_key_padding_mask=src_key_padding_mask) # true = ignore

        
        mask_float = (~src_key_padding_mask).unsqueeze(-1).float() # (B, S, 1), True=valid -> 1.0, False=ignore -> 0.0
        masked_sum = (transformer_output * mask_float).sum(dim=1) # (B, D)
        num_valid = mask_float.sum(dim=1).clamp(min=1e-9) # (B, 1)
        pooled_output = masked_sum / num_valid # (B, D)

         # Attention pooling: learn weights for each time step
        # Create a learnable parameter to score each time step
        # attn_weights = torch.softmax(torch.mean(transformer_output, dim=-1), dim=1)  # (B, S)
        # # Expand dims to multiply: (B, S, 1)
        # attn_weights = attn_weights.unsqueeze(-1)
        # pooled_output = torch.sum(transformer_output * attn_weights, dim=1)  # (B, d_model)
    
        return pooled_output


    def forward(self, t_seq, z_seq, v_seq, src_key_padding_mask):
        """
        Args:
            t_seq: (batch_size, seq_len) - Scaled time
            z_seq: (batch_size, seq_len) - Modality indices
            v_seq: (batch_size, seq_len) - Scaled values
            src_key_padding_mask: (batch_size, seq_len) - Bool mask (True=ignore)
        """
        # Get the CLS token representation
        rep = self.get_representation(t_seq, z_seq, v_seq, src_key_padding_mask)
        logits = self.classifier(rep)      
        
        return logits

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),            
            nn.Dropout(0.2),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, proj_dim)


        )
    def forward(self, x):
        return self.net(x)
    

