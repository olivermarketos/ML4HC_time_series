import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.optim as optim
from sklearn.metrics import  confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import random
import matplotlib.pyplot as plt
import time
import random
from tqdm import tqdm
import os
from Transformers import MedicalTimeSeriesDatasetTimeGrid,MedicalTimeSeriesDatasetTuple, collate_fn, TimeSeriesGridTransformer, TimeSeriesTupleTransformer, ContrastiveMedicalTimeSeriesDatasetTuple, augment_triplet, contrastive_collate_fn




def get_data_loaders(config):
    """Loads data and creates DataLoader objects."""
    
    data_path = config["data_dir"]

    y_train = np.load(os.path.join(data_path, 'outcomes_sorted_set-a.npy'), allow_pickle=True)
    y_val = np.load(os.path.join(data_path, 'outcomes_sorted_set-b.npy'),allow_pickle=True)
    y_test = np.load(os.path.join(data_path, 'outcomes_sorted_set-c.npy'), allow_pickle=True)

    if config["model_type"] == "time_grid":
        data_format = config["model_type"]
    elif config["model_type"] == "tuple" or config["model_type"] == "contrast" or config["model_type"] == "linear_probe":
        data_format = "tuple"
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")
    
    scaler_name = config["scaler_name"]
    X_train = np.load(os.path.join(data_path,f"{data_format}_processed_{scaler_name}_set-a.npy"), allow_pickle=True)
    X_val = np.load(os.path.join(data_path,f"{data_format}_processed_{scaler_name}_set-b.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(data_path,f"{data_format}_processed_{scaler_name}_set-c.npy"), allow_pickle=True)

  
    if config["model_type"] == "time_grid":
      # Feature dimension check
        config["grid_feature_dim"] = X_train.shape[2] # Update based on loaded data
        print(f"Grid Data Shapes: Train X: {X_train.shape}, Val X: {X_val.shape}, Test X: {X_test.shape}")

        train_dataset = MedicalTimeSeriesDatasetTimeGrid(X_train, y_train)
        val_dataset = MedicalTimeSeriesDatasetTimeGrid(X_val, y_val)
        test_dataset = MedicalTimeSeriesDatasetTimeGrid(X_test, y_test)
        collate_func = None # Default collate for grid data

    elif config["model_type"] == "tuple" or config["model_type"] == "linear_probe":
      
        # Modality count check (find max index + 1, assuming 0 is padding or valid)
        all_z = [t[1] for patient in X_train for t in patient]
        num_modalities = max(all_z) + 1 if all_z else 1
        if num_modalities > config["tuple_num_modalities"]:
             print(f"Warning: Found {num_modalities-1} as max modality index. Updating config.")
             config["tuple_num_modalities"] = num_modalities
        elif num_modalities < config["tuple_num_modalities"]:
             print(f"Warning: Max modality index is {num_modalities-1}, but config expects {config['tuple_num_modalities']-1}. Using config value.")

        print(f"Tuple Data: Train Patients: {len(X_train)}, Val Patients: {len(X_val)}, Test Patients: {len(X_test)}")
        print(f"Using {config['tuple_num_modalities']} modalities (including padding index {config['PAD_INDEX_Z']}")


        train_dataset = MedicalTimeSeriesDatasetTuple(X_train, y_train, max_seq_len=config["tuple_max_seq_len"])
        val_dataset = MedicalTimeSeriesDatasetTuple(X_val, y_val, max_seq_len=config["tuple_max_seq_len"])
        test_dataset = MedicalTimeSeriesDatasetTuple(X_test, y_test, max_seq_len=config["tuple_max_seq_len"])
        collate_func = collate_fn # Use the custom collate_fn

    elif config["model_type"] == "contrast":
        contrastive_dataset = ContrastiveMedicalTimeSeriesDatasetTuple(
                X_train, y_train, augmentation=augment_triplet)
        
        contrastive_loader = DataLoader(
            contrastive_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            collate_fn=contrastive_collate_fn)

        return contrastive_loader, None, None
        
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    print(f"Labels - Train: {len(y_train)} (Positive: {y_train.sum()}), Val: {len(y_val)} (Positive: {y_val.sum()}), Test: {len(y_test)} (Positive: {y_test.sum()})")

    # --- Weighted Sampler for Imbalance ---
    class_counts = np.bincount(y_train.astype(int))
    class_weights = 1. / class_counts
    sample_weights = np.array([class_weights[int(t)] for t in y_train])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # --- DataLoaders ---
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler, collate_fn=collate_func)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"] * 2, shuffle=False, collate_fn=collate_func) # Larger batch size for validation
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"] * 2, shuffle=False, collate_fn=collate_func)

    print("DataLoaders created.")
    return train_loader, val_loader, test_loader

def get_model(config):
    """Initializes the model based on the specified type."""
    
    if config["model_type"] == "time_grid":
        model = TimeSeriesGridTransformer(
            feature_dim=config["grid_feature_dim"],
            d_model=config["grid_d_model"],
            nhead=config["grid_nhead"],
            num_layers=config["grid_num_layers"],
            dropout=config["grid_dropout"]
        )
    elif config["model_type"] == "tuple" or config["model_type"] == "contrast":
        model = TimeSeriesTupleTransformer(
            num_modalities=config["tuple_num_modalities"],
            d_model=config["tuple_d_model"],
            nhead=config["tuple_nhead"],
            num_encoder_layers=config["tuple_num_encoder_layers"],
            dim_feedforward=config["tuple_dim_feedforward"],
            num_classes=1, # Binary classification
            dropout=config["tuple_dropout"],
            max_seq_len=config["tuple_max_seq_len"],
            modality_emb_dim=config["tuple_modality_emb_dim"]
        )
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    model.to(config["device"])
    print(f"Model ({config['model_type']}) created and moved to {config['device']}.")
 
    return model

def get_optimizer(model, config):
    """Initializes the optimizer based on the specified type."""
    
    if config["optimizer"] == "adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    elif config["optimizer"] == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9, weight_decay=config["weight_decay"])
    elif config["optimizer"] == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")

    print(f"Optimizer ({config['optimizer']}) created.")
    return optimizer


def train_model(model,
                dataloader,
                criterion,
                optimizer,
                device,
                model_type,
                linear_probe=None,
                supervised=True,
                temperature=0.2):   
    
    model.train()
    if linear_probe:
        linear_probe.train()
    
    total_loss = 0.0
   
    for batch in tqdm(dataloader, desc="Training", leave=False):
        

        if model_type == "tuple":
            t_seq, z_seq, v_seq, attn_mask, labels = batch
            t_seq, z_seq, v_seq = t_seq.to(device), z_seq.to(device), v_seq.to(device)
            attn_mask, labels = attn_mask.to(device), labels.to(device)
            
            logits = model(t_seq, z_seq, v_seq, attn_mask) # Pass correct mask polarity
            y = labels # Ensure y is of shape (batch_size, 1)
            
        elif model_type == "time_grid":
            x, label = batch
        
            x = x.to(device)
            y = label.to(device)
            logits = model(x)

        elif model_type == "linear_probe":
            model.eval()
            t_seq, z_seq, v_seq, attn_mask, labels = batch
            t_seq, z_seq, v_seq, attn_mask, labels = t_seq.to(device), z_seq.to(device), v_seq.to(device), attn_mask.to(device), labels.to(device)
            with torch.no_grad():
                rep = model.get_representation(t_seq, z_seq, v_seq, attn_mask)
            
            try:
                preds = linear_probe(rep)
            except AttributeError:
                raise ValueError("Linear probe not initialized or not callable.")
            loss = criterion(preds, labels)

            y = labels
            logits = preds
        # elif model_type == "contrast" and projection_head:
        #     (t_seq1, z_seq1, v_seq1, attn_mask1), (t_seq2, z_seq2, v_seq2, attn_mask2), _ = batch
        #     t_seq1, z_seq1, v_seq1, attn_mask1 = t_seq1.to(device), z_seq1.to(device), v_seq1.to(device), attn_mask1.to(device)
        #     t_seq2, z_seq2, v_seq2, attn_mask2 = t_seq2.to(device), z_seq2.to(device), v_seq2.to(device), attn_mask2.to(device)

        #     rep1 = model.get_representation(t_seq1, z_seq1, v_seq1, attn_mask1)
        #     rep2 = model.get_representation(t_seq2, z_seq2, v_seq2, attn_mask2)

        #     proj1 = projection_head(rep1)
        #     proj2 = projection_head(rep2)

        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        optimizer.zero_grad()

        loss = criterion(logits, y)
        # else:
        #     loss = criterion(proj1, proj2, temperature)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        total_loss += loss.item()
   
    avg_loss = total_loss / len(dataloader)    
   
    return avg_loss


def evaluate(model, dataloader, criterion, device, model_type, linear_probe=None):
    model.eval()
    if linear_probe:
        linear_probe.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if model_type == "tuple":
                t_seq, z_seq, v_seq, attn_mask, labels = batch
                t_seq, z_seq, v_seq, attn_mask, labels = t_seq.to(device), z_seq.to(device), v_seq.to(device), attn_mask.to(device), labels.to(device)
                logits = model(t_seq, z_seq, v_seq, attn_mask)
                y = labels

                # Store original labels for metrics


            elif model_type == "time_grid":
                x, labels = batch
                x = x.to(device)
                y = labels.to(device)
                logits = model(x)

                # Store original labels for metrics

            elif model_type == "linear_probe":
                t_seq, z_seq, v_seq, attn_mask, labels = batch
                t_seq, z_seq, v_seq, attn_mask, labels = t_seq.to(device), z_seq.to(device), v_seq.to(device), attn_mask.to(device), labels.to(device)
                rep = model.get_representation(t_seq, z_seq, v_seq, attn_mask)
                logits = linear_probe(rep)
                y = labels

            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.flatten().flatten())
            all_labels.extend(y.cpu().numpy().flatten())

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auroc = roc_auc_score(all_labels, all_preds)
    auprc = average_precision_score(all_labels, all_preds)
    confusion_matrix_result = confusion_matrix(all_labels, (all_preds > 0.5).astype(int))
    return avg_loss, auroc, auprc, confusion_matrix_result
 