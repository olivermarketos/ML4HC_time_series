import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import csv


def convert_data_to_tensor(features, outcomes, device='cpu'):
    """
    Converts tabular time-series features and outcome labels into PyTorch tensors suitable for LSTM input.

    This function groups the input data by 'RecordID' to form sequences for each individual record,
    drops unnecessary columns ('RecordID' and 'Time'), and stacks them into a batch of sequences.
    Corresponding outcome labels are matched using the 'RecordID' index.

    Parameters:
    -----------
    features : pandas.DataFrame
        A DataFrame containing time-series feature values. Expected to include:
            - 'RecordID' column: identifies each individual sequence (e.g., a patient or entity).
            - 'Time' column: time ordering of observations (dropped during tensor creation).
            - Remaining columns: actual input features.

    outcomes : pandas.DataFrame
        A DataFrame containing binary or continuous outcome labels, indexed by 'RecordID'.
        Each unique 'RecordID' should appear once and map to a single label.

    device : str, optional (default='cpu')
        The device to move the output tensors to (e.g., 'cuda', 'mps', or 'cpu').

    Returns:
    --------
    features_batch : torch.Tensor
        A 3D tensor of shape (batch_size, seq_length, input_size), where:
            - batch_size: number of unique RecordIDs (i.e., sequences).
            - seq_length: number of time steps for each record (can vary if padded later).
            - input_size: number of input features per time step.

    targets_batch : torch.Tensor
        A 2D tensor of shape (batch_size, 1), containing one target label per sequence.
    """
    all_sequences = []
    all_targets = []
    
    for ID in features['RecordID'].unique():
        features_df = features.loc[features['RecordID'] == ID, :].drop(['RecordID', 'Time'], axis=1)
        features_tensor = torch.tensor(features_df.values, dtype=torch.float32)

        target = torch.tensor(outcomes.loc[ID].values).float()

        all_sequences.append(features_tensor)
        all_targets.append(target)
    
    features_batch = torch.stack(all_sequences)
    targets_batch = torch.stack(all_targets)
    
    return features_batch.to(device), targets_batch.to(device)

class Classifier(nn.Module):
    """
    A simple LSTM-based binary classifier.

    Args:
        input_size (int): Number of input features per time step.
        hidden_size (int): Number of features in the LSTM hidden state.
        bidirectional (bool, optional): If True, uses a bidirectional LSTM. Defaults to False.
    """
    def __init__(self, input_size, hidden_size, bidirectional=False):
        super(Classifier, self).__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional)
        self.classifier = nn.Linear(1, 1)  # Final linear layer expects 1-dimensional input

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (seq_len, batch_size, input_size).

        Returns:
            Tensor: Output logits of shape (batch_size, 1).
        """
        x, _ = self.lstm(x)  # Run input through LSTM
        x = x[:, -1, :]      # Take output from the last time step

        if self.bidirectional:
            # If bidirectional, average across hidden units and reshape to (batch_size, 1)
            x = x.mean(axis=1).reshape(4000, 1)

        return self.classifier(x)  # Pass through final linear layer

def train_model(model, criterion, X_train, y_train, X_val, y_val, initial_parameter=None, num_epochs=100):
    """
    Trains a PyTorch model using a given loss criterion and optimizer, and evaluates on a validation set.

    Args:
        model (nn.Module): The PyTorch model to train.
        criterion (nn.Module): Loss function, e.g., nn.BCEWithLogitsLoss().
        X_train (Tensor): Training input features (Tensor).
        y_train (Tensor): Training labels (Tensor).
        X_val (Tensor): Validation input features (Tensor).
        y_val (Tensor): Validation labels (Tensor).
        initial_parameter (float, optional): If specified, fills all model parameters with this value before training.
        num_epochs (int): Number of training epochs. Default is 100.

    Returns:
        Tuple:
            - model (nn.Module): The trained model.
            - metrics (dict): Dictionary containing training/validation losses, accuracies, AUROCs, and AUPRCs.
    """
    # Ensure labels are 1D
    y_train = y_train.squeeze()
    y_val = y_val.squeeze()
    
    # Convert validation labels to numpy once (for metric computation)
    y_val_numpy = y_val.cpu().numpy()
    
    # Metrics to collect during training
    train_losses = []
    val_losses = []
    accuracies = []
    aurocs = []
    auprcs = []

    # Optional: initialize all model parameters to a fixed value
    if initial_parameter != None:
        with torch.no_grad():
            for param in model.parameters():
                param.fill_(initial_parameter)
                param.requires_grad_(True)

    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters())

    # Training loop
    for epoch in tqdm(range(num_epochs)):
        model.train()
        y_pred = model(X_train).squeeze()  # Forward pass on training data

        loss = criterion(y_pred, y_train)  # Compute training loss
        train_losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation phase
        model.eval()
        y_pred = model(X_val).squeeze()  # Forward pass on validation data

        loss = criterion(y_pred, y_val)  # Compute validation loss
        val_losses.append(loss.item())

        # Compute metrics
        predictions = torch.sigmoid(y_pred).round().detach().cpu().numpy()
        accuracies.append(accuracy_score(y_val_numpy, predictions))
        aurocs.append(roc_auc_score(y_val_numpy, predictions))
        auprcs.append(average_precision_score(y_val_numpy, predictions))

        # Early stopping logic placeholder (commented out)
        # if best_val_loss - loss < 1e-3:
        #     break
        # else:
        #     best_val_loss = loss
        
    return model, {
        'train losses': train_losses,
        'val losses': val_losses,
        'accuracies': accuracies,
        'aurocs': aurocs,
        'auprcs': auprcs
    }

def test_model(model, criterion, X_test, y_test):
    """
    Evaluates a trained model on a test dataset using various binary classification metrics.

    Args:
        model (nn.Module): The trained PyTorch model.
        criterion (nn.Module): Loss function used during evaluation, e.g., nn.BCEWithLogitsLoss().
        X_test (Tensor): Test input features.
        y_test (Tensor): Test labels.

    Returns:
        Tuple:
            - loss (Tensor): Loss value on the test set.
            - accuracy (float): Classification accuracy.
            - auroc (float): Area under the ROC curve.
            - auprc (float): Area under the precision-recall curve.
            - conf_matrix (ndarray): Confusion matrix as a NumPy array.
    """
    # Ensure labels are 1D
    y_test = y_test.squeeze()
    
    # Store a NumPy version of the ground truth labels for metric computation
    y_test_numpy = y_test.cpu().numpy()

    model.eval()  # Set model to evaluation mode

    with torch.no_grad():  # Disable gradient computation for evaluation
        y_pred = model(X_test).squeeze()  # Forward pass on test data

        loss = criterion(y_pred, y_test)  # Compute test loss

        # Apply sigmoid to logits, round to binary predictions, convert to NumPy
        predictions = torch.sigmoid(y_pred).round().cpu().numpy()

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test_numpy, predictions)
        auroc = roc_auc_score(y_test_numpy, predictions)
        auprc = average_precision_score(y_test_numpy, predictions)
        conf_matrix = confusion_matrix(y_test_numpy, predictions)

    return loss, accuracy, auroc, auprc, conf_matrix


def main():
    # Setting the device to GPU if possible
    device = None
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    # Loading the data
    data_train = pd.read_parquet('./data_processed/set-a.parquet')
    data_validate = pd.read_parquet('data_processed/set-b.parquet')
    data_test = pd.read_parquet('./data_processed/set-c.parquet')
    
    # Loading the outcome data
    outcomes_train = pd.read_csv('./data/Outcomes-a.txt')
    outcomes_train = outcomes_train.loc[:, ['RecordID', 'In-hospital_death']].set_index('RecordID')
    outcomes_validate = pd.read_csv('./data/Outcomes-b.txt')
    outcomes_validate = outcomes_validate.loc[:, ['RecordID', 'In-hospital_death']].set_index('RecordID')
    outcomes_test = pd.read_csv('./data/Outcomes-c.txt')
    outcomes_test = outcomes_test.loc[:, ['RecordID', 'In-hospital_death']].set_index('RecordID')
    
    # Scaling the data with min-max scaler
    scaler = MinMaxScaler()
    data_train_scaled = data_train.copy()
    data_train_scaled[data_train_scaled.columns.drop(['RecordID', 'Time'])] = scaler.fit_transform(data_train[data_train.columns.drop(['RecordID', 'Time'])])

    data_val_scaled = data_validate.copy()
    data_val_scaled[data_val_scaled.columns.drop(['RecordID', 'Time'])] = scaler.fit_transform(data_validate[data_validate.columns.drop(['RecordID', 'Time'])])

    data_test_scaled = data_test.copy()
    data_test_scaled[data_test_scaled.columns.drop(['RecordID', 'Time'])] = scaler.fit_transform(data_test[data_test.columns.drop(['RecordID', 'Time']).values])
    
    # Creating tensors from the data for the network
    X_train, y_train = convert_data_to_tensor(data_train_scaled, outcomes_train, device)
    X_val, y_val = convert_data_to_tensor(data_val_scaled, outcomes_validate, device)
    X_test, y_test = convert_data_to_tensor(data_test_scaled, outcomes_test, device)
    
    # Initializing the parameters of the model and the number of epochs
    input_size = 41
    hidden_size = 1
    initial_parameter =.1
    num_epochs = 1000
    
    # Creating the model and the loss function
    model = Classifier(input_size, hidden_size).to(device)
    model_bi = Classifier(input_size, hidden_size, bidirectional=True).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=(y_train.shape[0] - y_train.sum()) / y_train.sum())
    
    # Training the models
    trained_model, metrics = train_model(model, criterion, X_train, y_train, X_val, y_val, initial_parameter, num_epochs)
    trained_model_bi, metrics_bi = train_model(model_bi, criterion, X_train, y_train, X_val, y_val, initial_parameter, num_epochs)
    
    # Saving the models
    torch.save(trained_model, './trained_models/lstm_unidirectional.pth')
    torch.save(trained_model_bi, './trained_models/lstm_bidirectional.pth')
    
    # Testing the models
    test_loss, test_accuracy, test_auroc, test_auprc, test_conf_matrix = test_model(trained_model, criterion, X_test, y_test)
    test_loss_bi, test_accuracy_bi, test_auroc_bi, test_auprc_bi, test_conf_matrix_bi = test_model(trained_model_bi, criterion, X_test, y_test)
    
    # Saving the test set results to files
    metrics = {
        'loss': test_loss.item(),
        'accuracy': test_accuracy,
        'auROC': test_auroc,
        'auPRC': test_auprc
    }
    metrics_bi = {
        'loss': test_loss_bi.item(),
        'accuracy': test_accuracy_bi,
        'auROC': test_auroc_bi,
        'auPRC': test_auprc_bi
    }
    
    with open('./results/lstm_unidirectional_results.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())

        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(metrics)
    
    with open('./results/lstm_bidirectional_results.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=metrics_bi.keys())

        if file.tell() == 0:
            writer.writeheader()

        writer.writerow(metrics_bi)
        
    # Saving confusion matrices
    ConfusionMatrixDisplay(test_conf_matrix).plot()
    plt.title('Unidirectional LSTM')
    plt.savefig('./results/lstm_unidirectional_conf_matrix.png')
    plt.clf()
    
    ConfusionMatrixDisplay(test_conf_matrix_bi).plot()
    plt.title('Bidirectional LSTM')
    plt.savefig('./results/lstm_bidirectional_conf_matrix.png')
    
if __name__ == "__main__":
    main()
    