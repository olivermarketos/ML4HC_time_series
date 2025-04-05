import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, f1_score, precision_score, recall_score
)
import time
import os
import random
from tqdm.auto import tqdm # For progress bars
import json # For logging results
import datetime # For timestamping logs

# --- Import your model definitions ---
# Assume transformers.py is in the same directory or accessible
from transformers import (
    TimeSeriesGridTransformer,
    TimeSeriesTupleTransformer,
    MedicalTimeSeriesDatasetTimeGrid,
    MedicalTimeSeriesDatasetTuple,
    collate_fn,
    PAD_INDEX_Z,
    MAX_SEQ_LEN
)

# --- Configuration ---
config = {
    "model_type": "tuple",  # 'grid' or 'tuple'
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "epochs": 20,
    "batch_size": 64,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "patience": 5, # For early stopping based on validation performance

    # --- Paths (MODIFY THESE) ---
    "data_dir": "path/to/your/processed_data",
    "output_dir": "output", # General output directory
    # --- New Paths for Persistent Tracking ---
    "best_score_file": "output/best_test_auroc.json",
    "best_model_path": "output/best_overall_model.pth", # Stores only the overall best model
    "results_log_file": "output/run_results_log.jsonl", # Log of all runs
    # --- Primary metric for comparison ---
     "primary_metric": "auroc", # 'auroc' or 'auprc' or 'f1_optimal'

    # Model Specific Hyperparameters
    "grid_d_model": 128,
    "grid_nhead": 4,
    "grid_num_layers": 2,
    "grid_dropout": 0.2,
    "grid_feature_dim": 41,

    "tuple_d_model": 128,
    "tuple_nhead": 4,
    "tuple_num_encoder_layers": 2,
    "tuple_dim_feedforward": 256,
    "tuple_dropout": 0.2,
    "tuple_num_modalities": 42,
    "tuple_modality_emb_dim": 64,
    "tuple_max_seq_len": MAX_SEQ_LEN,
}

# --- Utility Functions ---
def set_seed(seed):
    # ... (implementation as before)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(config):
    # ... (implementation mostly as before) ...
    print("Loading data...")
    data_path = config["data_dir"]
    data_format = config["model_type"]

    # Load features (X)
    X_train = np.load(os.path.join(data_path, f"{data_format}_processed_set-a.npy"), allow_pickle=True)
    X_val = np.load(os.path.join(data_path, f"{data_format}_processed_set-b.npy"), allow_pickle=True)
    X_test = np.load(os.path.join(data_path, f"{data_format}_processed_set-c.npy"), allow_pickle=True)

    # Load labels (y)
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    y_val = np.load(os.path.join(data_path, 'y_val.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    if config["model_type"] == "grid":
        config["grid_feature_dim"] = X_train.shape[2]
        print(f"Grid Data Shapes: Train X: {X_train.shape}, Val X: {X_val.shape}, Test X: {X_test.shape}")
        train_dataset = MedicalTimeSeriesDatasetTimeGrid(X_train, y_train)
        val_dataset = MedicalTimeSeriesDatasetTimeGrid(X_val, y_val)
        test_dataset = MedicalTimeSeriesDatasetTimeGrid(X_test, y_test)
        collate_func = None

    elif config["model_type"] == "tuple":
        # Modality count check (simplified)
        all_z = [t[1] for patient in X_train for t in patient]
        num_modalities = max(all_z) + 1 if all_z else 1
        if num_modalities > config["tuple_num_modalities"]:
             print(f"Warning: Found {num_modalities-1} as max modality index. Updating config.")
             config["tuple_num_modalities"] = num_modalities
        elif num_modalities < config["tuple_num_modalities"]:
             print(f"Warning: Max modality index is {num_modalities-1}, but config expects {config['tuple_num_modalities']-1}. Using config value.")

        print(f"Tuple Data: Train Patients: {len(X_train)}, Val Patients: {len(X_val)}, Test Patients: {len(X_test)}")
        print(f"Using {config['tuple_num_modalities']} modalities (including padding index {PAD_INDEX_Z})")

        train_dataset = MedicalTimeSeriesDatasetTuple(X_train, y_train, max_seq_len=config["tuple_max_seq_len"])
        val_dataset = MedicalTimeSeriesDatasetTuple(X_val, y_val, max_seq_len=config["tuple_max_seq_len"])
        test_dataset = MedicalTimeSeriesDatasetTuple(X_test, y_test, max_seq_len=config["tuple_max_seq_len"])
        collate_func = collate_fn

    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    print(f"Labels - Train: {len(y_train)} (Positive: {y_train.sum()}), Val: {len(y_val)} (Positive: {y_val.sum()}), Test: {len(y_test)} (Positive: {y_test.sum()})")

    # Weighted Sampler
    class_counts = np.bincount(y_train.astype(int))
    if len(class_counts) < 2:
        print("Warning: Only one class present in training data. WeightedRandomSampler may not work as expected.")
        sampler = None # Fallback to default sampler
    else:
        class_weights = 1. / class_counts
        sample_weights = np.array([class_weights[int(t)] for t in y_train])
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=sampler, collate_fn=collate_func, num_workers=2, pin_memory=True, drop_last=True) # drop_last added
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"] * 2, shuffle=False, collate_fn=collate_func, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"] * 2, shuffle=False, collate_fn=collate_func, num_workers=2, pin_memory=True)

    print("DataLoaders created.")
    return train_loader, val_loader, test_loader

def get_model(config):
    # ... (implementation as before) ...
    if config["model_type"] == "grid":
        model = TimeSeriesGridTransformer(
            feature_dim=config["grid_feature_dim"],
            d_model=config["grid_d_model"],
            nhead=config["grid_nhead"],
            num_layers=config["grid_num_layers"],
            dropout=config["grid_dropout"]
        )
    elif config["model_type"] == "tuple":
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

def train_epoch(model, dataloader, criterion, optimizer, device, model_type):
    # ... (implementation using y_target_for_loss as corrected before) ...
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        optimizer.zero_grad()
        y_target_for_loss = None

        if model_type == 'grid':
            X_batch, y_batch = batch # y_batch is [B, 1]
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            y_target_for_loss = y_batch.float()

        elif model_type == 'tuple':
            t_seq, z_seq, v_seq, attn_mask, labels = batch # labels is [B, 1]
            t_seq, z_seq, v_seq, attn_mask, labels = t_seq.to(device), z_seq.to(device), v_seq.to(device), attn_mask.to(device), labels.to(device)
            logits = model(t_seq, z_seq, v_seq, attn_mask)
            y_target_for_loss = labels.float()

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if y_target_for_loss is None:
             raise RuntimeError("y_target_for_loss was not assigned in train loop.")

        loss = criterion(logits, y_target_for_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def evaluate(model, dataloader, criterion, device, model_type, return_preds=False):
    # ... (implementation using y_target_for_loss and careful metric calculation as corrected) ...
    model.eval()
    total_loss = 0.0
    all_preds_list = []
    all_labels_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            y_target_for_loss = None
            current_labels_for_list = None

            if model_type == "tuple":
                t_seq, z_seq, v_seq, attn_mask, labels = batch
                t_seq, z_seq, v_seq, attn_mask, labels = t_seq.to(device), z_seq.to(device), v_seq.to(device), attn_mask.to(device), labels.to(device)
                logits = model(t_seq, z_seq, v_seq, attn_mask)
                y_target_for_loss = labels.float()
                current_labels_for_list = labels # Shape [B, 1]

            elif model_type == "time_grid":
                x, labels = batch
                x = x.to(device)
                labels = labels.to(device)
                logits = model(x)
                y_target_for_loss = labels.float()
                current_labels_for_list = labels # Shape [B, 1]

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            if y_target_for_loss is None:
                 raise RuntimeError("y_target_for_loss was not assigned in evaluate loop.")
            loss = criterion(logits, y_target_for_loss)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds_list.extend(probs.flatten())
            all_labels_list.extend(current_labels_for_list.squeeze().cpu().numpy().flatten()) # Squeeze [B,1] -> [B]

    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds_list)
    all_labels = np.array(all_labels_list)

    results = {"loss": avg_loss}
    # Calculate metrics safely
    if len(all_labels) > 0 and len(np.unique(all_labels)) > 1:
        try: results["auroc"] = roc_auc_score(all_labels, all_preds)
        except ValueError: results["auroc"] = 0.5
        try: results["auprc"] = average_precision_score(all_labels, all_preds)
        except ValueError: results["auprc"] = np.mean(all_labels)
    else:
        results["auroc"] = 0.5
        results["auprc"] = 0.0

    # Return predictions if needed for threshold finding
    if return_preds:
        return results, all_labels, all_preds
    else:
        return results # Return dict

# --- New Helper Function ---
def find_optimal_threshold(labels, preds, metric='f1'):
    """Finds the optimal threshold based on the validation set predictions."""
    precisions, recalls, thresholds = precision_recall_curve(labels, preds)

    if metric == 'f1':
        # Handle division by zero and threshold array length
        f1_scores = [2 * (p * r) / (p + r) if (p + r) > 0 else 0
                     for p, r in zip(precisions[:-1], recalls[:-1])]
        if not f1_scores: return 0.5, 0.0 # Fallback if no scores
        best_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[best_idx]
        best_score = f1_scores[best_idx]

    # Add other metric options here if needed (e.g., Youden's J)
    # elif metric == 'youden':
    #    # Calculate TPR (Recall) and FPR
    #    # ... find threshold maximizing TPR - FPR ...

    else:
        raise ValueError(f"Unsupported metric for threshold finding: {metric}")

    print(f"Optimal threshold based on Val {metric.upper()}: {optimal_threshold:.4f} (Score: {best_score:.4f})")
    return optimal_threshold, best_score


# --- Main Execution ---
if __name__ == "__main__":
    set_seed(config["seed"])
    os.makedirs(config["output_dir"], exist_ok=True)

    # --- Load Previous Best Score ---
    best_score_so_far = -1.0
    metric_name = config["primary_metric"] # e.g., 'auroc'
    if os.path.exists(config["best_score_file"]):
        try:
            with open(config["best_score_file"], 'r') as f:
                data = json.load(f)
                if data.get("metric_name") == metric_name: # Ensure metric matches
                    best_score_so_far = data.get("score", -1.0)
                    print(f"Loaded previous best {metric_name}: {best_score_so_far:.4f}")
                else:
                    print(f"Warning: Metric in {config['best_score_file']} ({data.get('metric_name')}) doesn't match current metric ({metric_name}). Starting fresh.")
        except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
            print(f"Could not load or parse best score file: {e}. Starting fresh.")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # 2. Initialize Model, Criterion, Optimizer
    model = get_model(config)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=config["patience"] // 2, factor=0.5, verbose=True) # Schedule based on validation metric

    # 3. Training Loop
    best_val_metric = -1.0 # Track best validation metric for *this run*
    epochs_no_improve = 0
    train_model_path = os.path.join(config["output_dir"], f"model_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth") # Unique path for this run's best epoch model

    print("\n--- Starting Training ---")
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config["device"], config["model_type"])
        val_results = evaluate(model, val_loader, criterion, config["device"], config["model_type"])
        val_metric = val_results.get(metric_name, 0.0) # Use primary metric for validation check

        print(f"Epoch {epoch+1} Results - Train Loss: {train_loss:.4f}, Val Loss: {val_results['loss']:.4f}, Val {metric_name.upper()}: {val_metric:.4f}")

        scheduler.step(val_metric) # Scheduler step based on validation metric

        # Checkpointing based on validation performance for *this run*
        if val_metric > best_val_metric:
            print(f"Validation {metric_name.upper()} improved ({best_val_metric:.4f} -> {val_metric:.4f}). Saving model for this run to {train_model_path}...")
            best_val_metric = val_metric
            torch.save(model.state_dict(), train_model_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"Validation {metric_name.upper()} did not improve. ({epochs_no_improve}/{config['patience']})")

        if epochs_no_improve >= config["patience"]:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print("\n--- Training Finished ---")

    # 4. Load Best Model from *this run* (based on validation) for final evaluation
    print(f"\n--- Loading best model from this run ({train_model_path}) for final evaluation ---")
    if os.path.exists(train_model_path):
        model.load_state_dict(torch.load(train_model_path, map_location=config["device"]))
        print("Best model from this run loaded successfully.")
    else:
        print("Warning: No model checkpoint found for this run. Evaluating the final state.")

    # 5. Evaluate on Validation Set to find Optimal Threshold
    print("\n--- Evaluating on Validation Set to Find Optimal Threshold ---")
    val_results, val_labels, val_preds = evaluate(model, val_loader, criterion, config["device"], config["model_type"], return_preds=True)
    optimal_threshold, _ = find_optimal_threshold(val_labels, val_preds, metric='f1') # Find threshold using F1 on validation

    # 6. Evaluate on Test Set using the Best Model from this run
    print("\n--- Evaluating on Test Set ---")
    test_results, test_labels, test_preds = evaluate(model, test_loader, criterion, config["device"], config["model_type"], return_preds=True)
    current_test_primary_metric = test_results.get(metric_name, 0.0)
    print(f"\nTest Results (Default Threshold 0.5): Loss={test_results['loss']:.4f}, AuROC={test_results.get('auroc', 0.0):.4f}, AuPRC={test_results.get('auprc', 0.0):.4f}")

    # 7. Calculate Test Metrics with Optimal Threshold
    test_preds_optimal = (test_preds > optimal_threshold).astype(int)
    test_results_optimal = {}
    if len(test_labels) > 0 and len(np.unique(test_labels)) > 1:
        test_results_optimal['f1_optimal'] = f1_score(test_labels, test_preds_optimal)
        test_results_optimal['precision_optimal'] = precision_score(test_labels, test_preds_optimal)
        test_results_optimal['recall_optimal'] = recall_score(test_labels, test_preds_optimal)
        try:
            cm = confusion_matrix(test_labels, test_preds_optimal)
            test_results_optimal['confusion_matrix_optimal'] = cm.tolist() # Convert to list for JSON
        except ValueError:
             test_results_optimal['confusion_matrix_optimal'] = None
    else:
        test_results_optimal['f1_optimal'] = 0.0
        test_results_optimal['precision_optimal'] = 0.0
        test_results_optimal['recall_optimal'] = 0.0
        test_results_optimal['confusion_matrix_optimal'] = None

    print(f"Test Results (Optimal Threshold {optimal_threshold:.4f}): F1={test_results_optimal['f1_optimal']:.4f}, Precision={test_results_optimal['precision_optimal']:.4f}, Recall={test_results_optimal['recall_optimal']:.4f}")
    if test_results_optimal['confusion_matrix_optimal']:
        print(f"Confusion Matrix (Optimal Threshold):\n{np.array(test_results_optimal['confusion_matrix_optimal'])}")

    # Combine all results for logging
    final_results_log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config, # Log the config used for this run
        "best_val_metric": best_val_metric,
        "test_results_default_thresh": test_results,
        "optimal_threshold": optimal_threshold,
        "test_results_optimal_thresh": test_results_optimal,
        "run_model_path": train_model_path # Path to the model saved for this specific run
    }

    # --- Comparison and Saving Logic ---
    print(f"\n--- Comparing with Best Overall {metric_name.upper()} ({best_score_so_far:.4f}) ---")
    # Use the primary metric from the *default* threshold test results for comparison
    # Or potentially use f1_optimal if that's preferred: current_test_primary_metric = test_results_optimal.get('f1_optimal', 0.0)

    save_best = False
    if current_test_primary_metric > best_score_so_far:
        print(f"*** New best {metric_name.upper()} found! ({current_test_primary_metric:.4f} > {best_score_so_far:.4f}) ***")
        best_score_so_far = current_test_primary_metric
        save_best = True

        # Save the new best score to file
        try:
            with open(config["best_score_file"], 'w') as f:
                json.dump({"metric_name": metric_name, "score": best_score_so_far}, f, indent=4)
            print(f"Updated best score file: {config['best_score_file']}")
        except IOError as e:
            print(f"Error saving best score file: {e}")

        # Save the current model as the new best *overall* model
        try:
            torch.save(model.state_dict(), config["best_model_path"])
            print(f"Saved new best overall model to: {config['best_model_path']}")
        except IOError as e:
            print(f"Error saving best overall model: {e}")
    else:
        print(f"Current test {metric_name.upper()} ({current_test_primary_metric:.4f}) did not exceed best overall score ({best_score_so_far:.4f}). Best model not updated.")

    # --- Always Log Results of This Run ---
    try:
        with open(config["results_log_file"], 'a') as f:
            json.dump(final_results_log, f)
            f.write('\n') # Add newline for JSON Lines format
        print(f"Appended results of this run to log file: {config['results_log_file']}")
    except IOError as e:
        print(f"Error appending results to log file: {e}")

    print("\n--- Run Finished ---")