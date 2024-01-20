# imports from installed libraries
import os
import numpy as np
import random
import torch
from torch import nn
from distutils.version import LooseVersion as Version


def set_all_seeds(seed):
    """
    Set the seed for all relevant RNGs to ensure reproducibility across runs.

    This function sets a fixed seed for random number generators in os, random,
    numpy, and torch, ensuring that the same sequences of random numbers will be
    generated across different program executions when the same seed is used. It is
    particularly useful when trying to reproduce results in machine learning experiments.

    credits: Sebastian Raschka
    """

    # Set the seed for generating random numbers in Python's os module
    os.environ["PL_GLOBAL_SEED"] = str(seed)

    # Set the seed for the default Python RNG
    random.seed(seed)

    # Set the seed for numpy's RNG
    np.random.seed(seed)

    # Set the seed for PyTorch's RNG
    torch.manual_seed(seed)

    # Ensure that CUDA kernels' randomness is also seeded if available
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    """
    Enforces deterministic behavior in PyTorch operations to ensure reproducibility.

    This function configures PyTorch to behave deterministically, especially when running
    on a CUDA (GPU) environment. It disables certain optimizations that introduce non-determinism,
    making sure that the same inputs across different runs produce the same outputs.

    Note: Some PyTorch operations do not support deterministic mode, and using this function
    may have performance implications due to disabled optimizations.

    credits: Sebastian Raschka
    """

    # If CUDA (GPU support) is available, set related options for deterministic behavior
    if torch.cuda.is_available():
        # Disable the auto-tuner that finds the best algorithm for a specific input configuration.
        # This is necessary for reproducibility as different algorithms might produce slightly different results.
        torch.backends.cudnn.benchmark = False

        # Enable CuDNN deterministic mode. This ensures that convolution operations are deterministic.
        torch.backends.cudnn.deterministic = True

    # Set the deterministic flag based on the version of PyTorch.
    # Different versions of PyTorch use different functions to enforce deterministic algorithms.
    if torch.__version__ <= Version("1.7"):
        # For versions 1.7 or older, use `torch.set_deterministic`
        torch.set_deterministic(True)
    else:
        # From version 1.8 forward, use `torch.use_deterministic_algorithms`
        torch.use_deterministic_algorithms(True)


def evaluate_epoch_loss(model, data_loader, device, criterion):
    """
    Evaluate the loss of a model over an entire epoch.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        data_loader (torch.utils.data.DataLoader): DataLoader containing the dataset for evaluation.
        device (torch.device): The device to run the model on (e.g., 'cuda' or 'cpu').
        criterion (torch.nn.modules.loss._Loss): The loss function used for evaluation.
                                        Note that, losses such as nn.MSELoss, nn.BCELoss, nn.CrossEntropyLoss
                                        all by default take loss average over elements in the batch

    Returns:
        tuple: A tuple containing the total loss over all batches (curr_loss) and the average loss per batch (avg_loss).
    """

    # Initialize variables to accumulate loss
    curr_loss = 0.0
    avg_loss = 0.0

    # Disable gradient computation as we are only evaluating the model
    with torch.no_grad():
        # Iterate over batches of data provided by the data_loader
        for features, targets in data_loader:
            features = features.to(device)
            pred = model(features)
            # Compute the current batch's loss using the provided criterion
            loss = criterion(pred, targets)
            # Sum up the loss
            curr_loss += loss.item()

        avg_loss = curr_loss / len(data_loader)  # average of number of minibatches

        return curr_loss, avg_loss


def evaluate_epoch_metrics(model, data_loader, device):
    """
    Evaluator for classifiers.

    Computes the accuracy of a model over an entire epoch.

    The function evaluates the model on a dataset provided by a DataLoader without
    updating the model parameters (no gradient computation) to calculate accuracy.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The DataLoader providing the dataset.
        device (torch.device or str): The device to perform computations on. Example: 'cuda' or 'cpu'.

    Returns:
        float: The accuracy percentage of the model on the dataset.
    """

    # Initialize variables to correct predictions, and total examples
    correct_pred, num_examples = 0, 0

    # No gradient is needed for evaluation
    with torch.no_grad():
        for features, targets in data_loader:
            # Move features and targets to the appropriate device
            features, targets = features.to(device), targets.to(device)

            # Get model predictions and compute loss
            logits = model(features)

            # Convert logits to predicted labels
            _, predicted_labels = torch.max(logits, 1)

            # Update the counts for correct predictions and total examples processed
            correct_pred += (predicted_labels == targets).sum()
            num_examples += targets.size(0)

    # Calculate the accuracy across all examples
    accuracy = correct_pred.float() / num_examples * 100

    return accuracy


def get_predictions(model, dataloader, device):
    """
    Generate prediction of labels for the given data using model.

    Args:
        model (torch.nn.Module): Trained PyTorch model for predictions.
              the model output should be logits of shape (batch_size, num_classes)
        dataloader (DataLoader): A PyTorch DataLoader for which predictions are to be made.
        device (torch.device): The device to run the model on.

    Returns:
        Tuple of numpy arrays:
            input: say (N, C, H, W) if CNN model
            true labels (N,): numeric labels
            predicted labels (N,): numeric labels
    """
    model.eval()
    input_all, labels_all, predictions_all = [], [], []

    with torch.no_grad():
        for input, labels in dataloader:
            input, labels = input.to(device), labels.to(device)
            outputs = model(input)
            # outputs: logits (batch_size, num_classes)
            # torch.max: This function returns two values: the maximum value
            # itself and the index of where the maximum value occurred.
            _, predictions = torch.max(outputs, 1)

            input_all.extend(input.cpu())
            labels_all.extend(labels.cpu())
            predictions_all.extend(predictions.cpu())

    return (
        torch.stack(input_all).numpy(),
        torch.stack(labels_all).numpy(),
        torch.stack(predictions_all).numpy(),
    )
