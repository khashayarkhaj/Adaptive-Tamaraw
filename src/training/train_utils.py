# a set of helper functions used during training
# Standard library
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party scientific computing
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.autograd import Variable

# Scikit-learn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit


# Weights & Biases
import wandb

# Local imports - utilities
import utils.config_utils as cm
from utils.file_operations import save_file, predict_model_size

# Local imports - training
from training.config_manager import ConfigManager
from training.datasets import MultitaskDataset

# Local imports - models
from models.RF.RF_model import getRF
from models.RF.rf_train import rf_training_loop
from models.RF.rf_utils import load_and_modify_RF_model, RF_input_processor
from models.Tik_Tok.model import ConvNet
from models.WF_Transformer.common_layer import StepOpt
from models.WF_Transformer.UTransformer import UTransformer



def measure_inference_time(
    model: torch.nn.Module,
    test_x, 
    test_y,
    num_runs: int = 10,
    pre_processor=None,
    warm_up: int = 2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    batch_size: Optional[int] = None
) -> Tuple[float, float, float, float, List[float]]:
    """
    Measure model inference time with warm-up runs and statistical analysis.
    
    Args:
        model: PyTorch model to evaluate
        test_x: test tensors for evaluating inference time
        num_runs: Number of inference runs to measure
        warm_up: Number of warm-up runs before timing
        device: Device to run inference on ('cuda' or 'cpu')
        batch_size: If provided, will also calculate throughput (samples/second)
    
    Returns:
        Tuple containing:
        - mean_time: Average inference time per batch
        - std_time: Standard deviation of inference times
        - min_time: Minimum inference time observed
        - max_time: Maximum inference time observed
        - all_times: List of all measured times
    """
    model.eval()
    

    if pre_processor: # prepare the data before being fed into the model
        test_x, test_y = pre_processor(test_x, test_y)

    

    
    
    
    test_data = Data.TensorDataset(test_x, test_y)
    test_loader = Data.DataLoader(dataset=test_data, batch_size= batch_size, shuffle=False)
    batch_start = 0

    

    ### Evaluation Loop
    times = []
    for step, (tr_x, tr_y) in enumerate(tqdm(test_loader, desc=f'Evaluating model on {len(test_loader)} batches for time inference anlaysis')):
        if step > num_runs:
            break
        if device is not None:
            tr_x = Variable(tr_x.to(device))           
            tr_y = Variable(tr_y.to(device))

        if device == 'cuda':
                torch.cuda.synchronize()
        if step >= warm_up:
            start_time = time.perf_counter()
        test_output = model(tr_x)
        
        if device == 'cuda':
                torch.cuda.synchronize()
        
        if step >= warm_up:
            end_time = time.perf_counter()
            times.append(end_time - start_time)
        
        del test_output

    if len(times) > 0:
        return np.mean(times) / batch_size
    else:
        return -1



def evaluate_model(
    model: torch.nn.Module,
    logger,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    batch_size: int,
    actual_websites: list = None,
    evaluation_mode: str = 'validation',
    pre_processor: callable = None,
    to_cpu: bool = True,
    log_confusion_matrix_wandb: bool = False,
    return_confusion_matrix: bool = False,
    class_rankings: np.ndarray = None,
    top_ks: list = None,
    device: torch.device = None,
    multi_task: bool = False,
    test_y_task2: torch.Tensor = None,
    return_predictions: bool = False,
    save_best_model: bool = True,
):
    """
    Evaluate a PyTorch model on test data with support for multi-task learning,
    top-k accuracy, per-website statistics, and confusion matrix generation.

    Args:
        model: PyTorch neural network model to evaluate
        logger: Logger instance for recording evaluation information
        test_x: Input features tensor for test data
        test_y: Ground truth labels tensor for test data (primary task)
        batch_size: Number of samples per batch during evaluation
        actual_websites: Optional list mapping each sample to its source website
                        (useful when labels are cluster indices rather than actual websites)
        evaluation_mode: String identifier for the evaluation phase (e.g., 'validation', 'test')
        pre_processor: Optional function to transform data before model inference
                      Should accept (x, y) and return transformed (x, y)
        to_cpu: If True, moves model and data to CPU before evaluation
        log_confusion_matrix_wandb: If True, logs confusion matrix to Weights & Biases
        return_confusion_matrix: If True, computes and returns confusion matrix
        class_rankings: Array of shape [num_samples, num_classes] with class labels
                       sorted by predicted likelihood (for top-k accuracy)
        top_ks: List of k values for computing top-k accuracy (e.g., [1, 5, 10])
        device: PyTorch device for computation (e.g., torch.device('cuda:0'))
        multi_task: If True, expects model to output predictions for two tasks
        test_y_task2: Ground truth labels for second task (required if multi_task=True)
        return_predictions: If True, returns all predictions and labels in output dict
        save_best_model: Currently unused parameter (kept for backwards compatibility)

    Returns:
        dict: Dictionary containing evaluation metrics:
            - 'total_accuracy': Overall classification accuracy (float)
            - 'accuracies_top_k': List of top-k accuracies for each k in top_ks
            - 'conf_matrix': Confusion matrix array (if return_confusion_matrix=True)
            - 'class_accuracies': Per-class accuracy array (if return_confusion_matrix=True)
            - 'website_accuracies': Dict mapping websites to their accuracies (if actual_websites provided)
            - 'total_accuracy_second_task': Accuracy for second task (if multi_task=True)
            - 'all_preds': All predicted labels (if return_predictions=True)
            - 'all_labels': All true labels (if return_predictions=True)
    """
    
    # ============================================================================
    # Early Exit: Handle empty test data
    # ============================================================================
    if test_x is None or len(test_x) == 0:
        return {
            'total_accuracy': None,
            'accuracies_top_k': None,
            'conf_matrix': None,
            'class_accuracies': None,
            'website_accuracies': None,
            'total_accuracy_second_task': None
        }
    
    # ============================================================================
    # Setup: Model Mode and Multi-task Configuration
    # ============================================================================
    model.eval()
    
    # Configure multi-task mode if enabled
    if multi_task:
        if test_y_task2 is None:
            logger.info('Multi Task Learning is true, but labels of the second task are not given. '
                       'As a result the evaluation will not be in multi task mode.')
            multi_task = False
        if hasattr(model, 'enable_multi_task'):
            model.enable_multi_task()
    
    # ============================================================================
    # Data Preprocessing
    # ============================================================================
    if pre_processor:
        test_x, test_y = pre_processor(test_x, test_y)
        if multi_task:
            _, test_y_task2 = pre_processor(None, test_y_task2)
    
    # Move data to CPU if requested
    if to_cpu:
        test_x = test_x.cpu()
        test_y = test_y.cpu()
        model = model.cpu()
        if multi_task:
            test_y_task2 = test_y_task2.cpu()
    
    # ============================================================================
    # Initialize Tracking Variables
    # ============================================================================
    # Primary task metrics
    total_correct = 0
    total_samples = 0
    
    # Secondary task metrics
    total_correct_second_task = 0
    
    # Top-k accuracy tracking
    accuracies_top_k = []
    if top_ks is not None:
        total_corects_k = [0 for _ in top_ks]
        accuracies_top_k = [0 for _ in top_ks]
    
    # Prediction and label storage
    all_preds = []
    all_labels = []
    
    # ============================================================================
    # Per-Website Statistics Setup (Optional)
    # ============================================================================
    if actual_websites is not None:
        # Initialize counters for per-website statistics
        website_correct = {}  # Correct predictions per website
        website_total = {}    # Total instances per website
        
        # Initialize counters for each unique website
        unique_websites = list(set(actual_websites))
        for website in unique_websites:
            website_correct[website] = 0
            website_total[website] = 0
        
        # Create batch-aligned website indices for efficient lookup
        website_indices = []
        for batch_idx in range(0, len(test_x), batch_size):
            batch_end = min(batch_idx + batch_size, len(test_x))
            website_indices.append(actual_websites[batch_idx:batch_end])
    
    # ============================================================================
    # Create DataLoader
    # ============================================================================
    if not multi_task:
        test_data = Data.TensorDataset(test_x, test_y)
    else:
        test_data = MultitaskDataset(x=test_x, y1=test_y, y2=test_y_task2)
    
    test_loader = Data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    
    batch_start = 0  # Track position for top-k accuracy calculation
    
    # ============================================================================
    # Main Evaluation Loop
    # ============================================================================
    for step, training_resources in enumerate(tqdm(test_loader, 
                                                    desc=f'Evaluating model on {len(test_loader)} batches of {evaluation_mode} set')):
        # Unpack batch data
        if multi_task:
            tr_x, tr_y, tr_y2 = training_resources
        else:
            tr_x, tr_y = training_resources
        
        # Move batch to device if specified
        if device is not None and not to_cpu:
            tr_x = Variable(tr_x.to(device))
            tr_y = Variable(tr_y.to(device))
            if multi_task:
                tr_y2 = Variable(tr_y2.to(device))
        
        # ------------------------------------------------------------------------
        # Forward Pass
        # ------------------------------------------------------------------------
        if not multi_task:
            test_output = model(tr_x)
            test_output2 = None
        else:
            test_output, test_output2 = model(tr_x)
        
        # Move labels back to CPU for metric calculation
        tr_y = tr_y.cpu()
        
        # Get predictions and accuracy for primary task
        pred_y, accuracy = get_predictions(test_output, tr_y)
        
        # Get predictions for secondary task if multi-task
        if multi_task:
            tr_y2 = tr_y2.cpu()
            pred_y2, accuracy2 = get_predictions(test_output2, tr_y2)
        
        # ------------------------------------------------------------------------
        # Store Predictions (if needed for confusion matrix or return)
        # ------------------------------------------------------------------------
        if return_confusion_matrix or log_confusion_matrix_wandb or return_predictions:
            all_preds.append(pred_y)
            all_labels.append(tr_y.numpy())
        
        # ------------------------------------------------------------------------
        # Update Per-Website Statistics
        # ------------------------------------------------------------------------
        if actual_websites is not None:
            batch_websites = website_indices[step]
            
            for i, (pred, true, website) in enumerate(zip(pred_y, tr_y.numpy(), batch_websites)):
                website_total[website] += 1
                if pred == true:
                    website_correct[website] += 1
        
        # ------------------------------------------------------------------------
        # Update Overall Metrics
        # ------------------------------------------------------------------------
        batch_size_actual = tr_y.size(0)
        total_samples += batch_size_actual
        total_correct += (pred_y == tr_y.numpy()).sum()
        
        if multi_task:
            total_correct_second_task += (pred_y2 == tr_y2.numpy()).sum()
        
        # Free memory
        del test_output, test_output2
        
        # ------------------------------------------------------------------------
        # Update Top-K Accuracy Metrics
        # ------------------------------------------------------------------------
        if top_ks is not None:
            batch_end = batch_start + batch_size_actual
            
            for idx, top_k in enumerate(top_ks):
                # Get top-k predicted classes for this batch
                topk_real_classes = class_rankings[batch_start:batch_end, :top_k]
                
                # Check if true label is in top-k predictions
                pred_y_expanded = pred_y.reshape(-1, 1)
                correct = np.any(pred_y_expanded == topk_real_classes, axis=1)
                total_corects_k[idx] += correct.sum()
            
            batch_start = batch_end
    
    # ============================================================================
    # Calculate Final Metrics
    # ============================================================================
    
    # Overall accuracy for primary task
    total_accuracy = total_correct * 1.0 / total_samples
    
    # Overall accuracy for secondary task (if applicable)
    total_accuracy_second_task = None
    if multi_task:
        total_accuracy_second_task = total_correct_second_task * 1.0 / total_samples
    
    # Per-website accuracies
    website_accuracies = None
    if actual_websites is not None:
        website_accuracies = {}
        for website in unique_websites:
            if website_total[website] > 0:
                website_accuracies[website] = website_correct[website] / website_total[website]
            else:
                website_accuracies[website] = 0.0
    
    # Top-k accuracies
    if top_ks is not None:
        for idx, top_k in enumerate(top_ks):
            accuracies_top_k[idx] = total_corects_k[idx] * 1.0 / total_samples
    
    # Concatenate all predictions and labels if needed
    if return_confusion_matrix or log_confusion_matrix_wandb or return_predictions:
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
    
    # ============================================================================
    # Confusion Matrix and Per-Class Accuracy
    # ============================================================================
    conf_matrix = None
    class_accuracies = None
    
    if return_confusion_matrix:
        # Compute confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        logger.info('Confusion Matrix:')
        logger.info(conf_matrix)
        
        # Calculate per-class accuracy
        class_accuracies = per_class_accuracy(conf_matrix)
        
        # Log per-class accuracies
        for i, accuracy in enumerate(class_accuracies):
            print(f"Accuracy for class {i}: {accuracy:.4f}")
    
    # ============================================================================
    # Weights & Biases Logging
    # ============================================================================
    if log_confusion_matrix_wandb:
        num_classes = len(np.unique(test_y.numpy()))
        class_names = [f'class {i}' for i in range(num_classes)]
        
        wandb.log({
            "Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=all_labels,
                preds=all_preds,
                class_names=class_names
            )
        })
    
    # ============================================================================
    # Cleanup and Return
    # ============================================================================
    # Disable multi-task mode if it was enabled
    if hasattr(model, 'disable_multi_task'):
        model.disable_multi_task()
    
    # Return comprehensive results dictionary
    return {
        'total_accuracy': total_accuracy,
        'accuracies_top_k': accuracies_top_k,
        'conf_matrix': conf_matrix,
        'class_accuracies': class_accuracies,
        'website_accuracies': website_accuracies,
        'total_accuracy_second_task': total_accuracy_second_task,
        'all_preds': all_preds if return_predictions else None,
        'all_labels': all_labels if return_predictions else None
    }
        


def get_predictions(output, true_y):
    """
    Calculates prediction accuracy from model outputs compared to true labels.
    
    Args:
        output (torch.Tensor): Model output tensor of shape (batch_size, num_classes) 
            containing class scores/logits. Must be on CPU or will be moved to CPU.
        true_y (torch.Tensor): Ground truth labels tensor of shape (batch_size,)
            containing integer class indices. Must be on CPU or will be moved to CPU.
    
    Returns:
        tuple: Contains:
            - pred_y (numpy.ndarray): Predicted class indices
            - accuracy (float): Fraction of correct predictions (between 0 and 1)
    """
    # Move tensors to CPU if they're on GPU
    output = output.cpu()
    true_y = true_y.cpu()
    
    pred_y = torch.max(output, 1)[1].data.numpy().squeeze()
    accuracy = (pred_y == true_y.numpy()).sum().item() * 1.0 / float(true_y.size(0))
    return pred_y, accuracy


def save_accuracy_plot(accuracies, n, save_path, k='', batch_size=None, learning_rate=None, optimizer=None, scheduler=None, validation_type = 'val', file_name = None):
    """
    Plot and save the top n accuracy for each epoch when training a wf classifier. Optional Additional parameters like batch size, learning rate, optimizer, and scheduler displayed within the plot area.
    Also displays the maximum accuracy value on the plot.

    Parameters:
        accuracies (list): List of accuracies for each epoch.
        n (int): Top n accuracy.
        save_path (str): Path to save the plot.
        k (int): k anonymity parameter.
        batch_size (int): Size of the batch used in training.
        learning_rate (float): Learning rate used in training.
        optimizer (str): Optimization method used.
        scheduler (str): Scheduler used alongside the optimizer.
    """
    epochs = list(range(1, len(accuracies) + 1))
    
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
    
    # Find maximum accuracy and its epoch
    max_accuracy = max(accuracies)
    max_epoch = accuracies.index(max_accuracy) + 1
    
    # Add marker and annotation for maximum value
    plt.plot(max_epoch, max_accuracy, 'ro', markersize=10)  # Red dot for maximum
    plt.annotate(f'Max: {max_accuracy:.2f}%', 
                xy=(max_epoch, max_accuracy),
                xytext=(10, 10),
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Basic plot setup
    title = f'Top {n} {validation_type} Accuracy per Epoch'
    if k != '':
        title += f' for K = {k}'
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(f'Top {n} Accuracy (%)')
    plt.grid(True)
    
    # Adding text inside the plot for training parameters
    params_text = ''
    if batch_size is not None and learning_rate is not None:
        params_text += f'Batch Size: {batch_size}\nLearning Rate: {learning_rate}\n'
    if optimizer is not None and scheduler is not None:
        params_text += f'Optimizer: {optimizer}\nScheduler: {scheduler}'
        save_path = os.path.join(save_path, f'bs_{batch_size}_lr_{learning_rate : .2f}_optimizer_{optimizer}_scheduler_{scheduler}')
    
    plt.text(0.02, 0.05, params_text, transform=plt.gca().transAxes, fontsize=9, 
             verticalalignment='bottom', 
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.5))
    
    if file_name is None:
        file_name = f'k={k}_top_{n}_accuracy_{validation_type}.png'
    save_file(dir_path=save_path, file_name= file_name)

def plot_train_metrics(training_loss, save_path, training_accuracy=None):
    # Ensure we have all required imports
    import matplotlib.pyplot as plt
    
    # Calculate min loss
    min_loss = min(training_loss)
    min_loss_epoch = training_loss.index(min_loss)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(training_loss, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    # Adjust y-axis limits to ensure annotation visibility
    y_min = min(training_loss) * 0.95  # 5% padding below minimum
    y_max = max(training_loss) * 1.15  # 15% padding above maximum
    plt.ylim(y_min, y_max)
    
    # Add annotation for minimum loss
    plt.annotate(f'Min Loss: {min_loss:.4f}',
                xy=(min_loss_epoch, min_loss),
                xytext=(10, 10),
                textcoords='offset points',
                ha='left',
                va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.legend()
    plt.grid(True)
    # Add padding to the plot
    plt.tight_layout(pad=2.0)
    save_file(dir_path=save_path, file_name='training_loss.png')
    
    # Plot training accuracy if it's not None
    if training_accuracy is not None:
        # Calculate max accuracy
        max_acc = max(training_accuracy)
        max_acc_epoch = training_accuracy.index(max_acc)
        
        plt.figure(figsize=(10, 6))
        plt.plot(training_accuracy, label='Training Accuracy', color='green')
        plt.title('Training Accuracy Over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        
        # Adjust y-axis limits to ensure annotation visibility
        acc_min = min(training_accuracy) * 0.95  # 5% padding below minimum
        acc_max = min(1.0, max(training_accuracy) * 1.15)  # 15% padding above maximum, capped at 1.0
        plt.ylim(acc_min, acc_max)
        
        # Add annotation for maximum accuracy
        plt.annotate(f'Max Accuracy: {max_acc:.2%}',
                    xy=(max_acc_epoch, max_acc),
                    xytext=(10, -10),
                    textcoords='offset points',
                    ha='left',
                    va='top',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.legend()
        plt.grid(True)
        # Add padding to the plot
        plt.tight_layout(pad=2.0)
        save_file(dir_path=save_path, file_name='training_accuracy.png')

# Example usage:
# plot_and_save_metrics([0.9, 0.8, 0.7], [0.6, 0.7, 0.8], '/path/to/save')




def train_wf_model(
    logger,  # Logger instance for tracking training progress
    num_classes: int,  # Number of classification classes
    hyperparam_manager,  # Configuration manager or wandb config containing hyperparameters
    training_loop: Callable,  # Function that executes one training epoch
    wf_model: Optional[torch.nn.Module] = None,  # Pre-initialized model (if None, will be created based on wf_model_type)
    input_processor: Optional[Callable] = None,  # Function to preprocess inputs before training
    save_model: bool = False,  # Whether to save the trained model
    train_dataset = None,  # Dataset object with .directions and .labels attributes
    test_dataset = None,  # Test dataset object
    val_data_set = None,  # Validation dataset object (optional)
    x_train: Optional[torch.Tensor] = None,  # Training features (alternative to train_dataset)
    y_train: Optional[torch.Tensor] = None,  # Training labels (alternative to train_dataset)
    x_test: Optional[torch.Tensor] = None,  # Test features
    y_test: Optional[torch.Tensor] = None,  # Test labels
    x_val: Optional[torch.Tensor] = None,  # Validation features
    y_val: Optional[torch.Tensor] = None,  # Validation labels
    cuda_id: Optional[int] = None,  # Specific CUDA device ID (if None, uses default GPU)
    use_wandb: bool = False,  # Whether to use Weights & Biases for experiment tracking
    class_rankings_val: Optional[List] = None,  # Pre-computed class rankings for validation top-k accuracy
    class_rankings_test: Optional[List] = None,  # Pre-computed class rankings for test top-k accuracy
    if_use_gpu: bool = False,  # Whether to use GPU for training
    topk_s: Optional[List[int]] = None,  # List of k values for top-k accuracy evaluation (e.g., [5, 10, 20])
    test_on_cpu: bool = False,  # Whether to move model to CPU during evaluation
    wf_model_type: Optional[str] = None,  # Model architecture type ('RF', 'WFT', 'tt')
    convert_data_to_torch: bool = False,  # Whether to convert numpy arrays to torch tensors
    use_input_processor: bool = True,  # Whether to apply input_processor to data
    report_train_accuracy: bool = False,  # Whether to compute training accuracy each epoch
    actual_websites_train: Optional[torch.Tensor] = None,  # Secondary labels for multi-task learning (training)
    actual_websites_val: Optional[torch.Tensor] = None,  # Secondary labels for multi-task learning (validation)
    actual_websites_test: Optional[torch.Tensor] = None,  # Secondary labels for multi-task learning (test)
    multi_task: bool = False,  # Whether to perform multi-task learning
    model_save_path: Optional[str] = None,  # Directory path to save the trained model
    should_evaluate: bool = True,  # Whether to run evaluation during training
    return_model: bool = False,  # Whether to return the trained model in results
    training_loop_tqdm: bool = True,  # Whether to show progress bar during training
    class_weights: Optional[torch.Tensor] = None,  # Weights for handling class imbalance
    val_acc_stop_threshold: Optional[float] = None  # Early stopping threshold (stop if val accuracy exceeds this)
) -> Dict[str, Any]:
    """
    Train a website fingerprinting (WF) classification model.
    
    This function handles the complete training pipeline including:
    - Data preparation and preprocessing
    - Model initialization
    - Training loop execution
    - Validation and testing
    - Metrics tracking and logging
    
    Args:
        logger: Logger instance for tracking training progress
        num_classes: Number of classification classes
        hyperparam_manager: Configuration manager or wandb config containing hyperparameters
                          Expected keys: 'train.num_epochs', 'train.batch_size', 
                          'train.learning_rate', 'train.lr_scheduler', 'train.optimizer'
        training_loop: Function that executes one training epoch
        wf_model: Pre-initialized PyTorch model (if None, created based on wf_model_type)
        input_processor: Function to preprocess inputs (x, y) -> (processed_x, processed_y)
        save_model: Whether to save the trained model weights
        train_dataset: Dataset object with .directions and .labels attributes
        test_dataset: Test dataset object
        val_data_set: Validation dataset (if None, test set is used for validation)
        x_train: Training features as tensor/array (alternative to train_dataset)
        y_train: Training labels as tensor/array
        x_test: Test features
        y_test: Test labels
        x_val: Validation features
        y_val: Validation labels
        cuda_id: Specific CUDA device ID (e.g., 0, 1) for multi-GPU systems
        use_wandb: Whether to use Weights & Biases for experiment tracking
        class_rankings_val: Pre-computed class rankings for validation top-k accuracy
        class_rankings_test: Pre-computed class rankings for test top-k accuracy
        if_use_gpu: Whether to attempt GPU usage
        topk_s: List of k values for top-k accuracy (e.g., [5, 10, 20])
        test_on_cpu: Whether to move model to CPU during evaluation (saves GPU memory)
        wf_model_type: Model architecture ('RF', 'WFT', 'tt')
        convert_data_to_torch: Convert numpy arrays to torch tensors
        use_input_processor: Whether to apply input preprocessing
        report_train_accuracy: Compute and log training accuracy each epoch
        actual_websites_train: Secondary task labels for multi-task learning (training)
        actual_websites_val: Secondary task labels for multi-task learning (validation)
        actual_websites_test: Secondary task labels for multi-task learning (test)
        multi_task: Enable multi-task learning mode
        model_save_path: Directory to save trained model weights
        should_evaluate: Whether to perform validation/test evaluation
        return_model: Include trained model in return dictionary
        training_loop_tqdm: Show progress bar during training epochs
        class_weights: Class weights for handling imbalanced datasets
        val_acc_stop_threshold: Early stopping threshold (stops if val acc > threshold)
    
    Returns:
        Dict containing training results:
            - 'val_accuracy_list': List of validation accuracies per epoch
            - 'val_accuracy_list_topks': List of top-k validation accuracies
            - 'test_accuracy': Final test accuracy
            - 'test_accuracy_top_ks': List of top-k test accuracies
            - 'confusion_matrix': Confusion matrix on test set
            - 'class_accuracies': Per-class accuracy breakdown
            - 'website_accuracies': Website-level accuracies (if applicable)
            - 'training_losses': List of training losses per epoch
            - 'training_accuracies': List of training accuracies per epoch (if computed)
            - 'model_size': Model size in bytes/parameters
            - 'total_training_time': Total training time in seconds
            - 'average_epoch_time': Average time per epoch in seconds
            - 'inference_time': Average inference time per sample (if evaluated)
            - 'model': Trained model (if return_model=True)
    """

    # ========================================================================
    # HYPERPARAMETER EXTRACTION
    # ========================================================================
    # Extract training hyperparameters from wandb config or custom config manager
    if use_wandb:
        wandb.init()
        hyperparam_manager = wandb.config  # Updated by sweep agent
        total_epochs = 100
        batch_size = hyperparam_manager.batch_size
        learning_rate = hyperparam_manager.learning_rate
        scheduler_method = hyperparam_manager.lr_scheduler
        optimization_method = hyperparam_manager.optimizer
    else:
        # Config manager with hierarchical keys like 'train.num_epochs'
        total_epochs = hyperparam_manager.get('train.num_epochs')
        batch_size = hyperparam_manager.get('train.batch_size')
        learning_rate = hyperparam_manager.get('train.learning_rate')
        scheduler_method = hyperparam_manager.get('train.lr_scheduler')
        optimization_method = hyperparam_manager.get('train.optimizer')
    
    logger.info(
        f"Training Configuration - Epochs: {total_epochs}, Batch Size: {batch_size}, "
        f"Learning Rate: {learning_rate}, LR Scheduler: {scheduler_method}, "
        f"Optimizer: {optimization_method}")

    # ========================================================================
    # DATA PREPARATION
    # ========================================================================
    # Extract features and labels from dataset objects if provided
    if train_dataset:
        x_train = train_dataset.directions
        y_train = train_dataset.labels
        x_test = test_dataset.directions
        y_test = test_dataset.labels
        
        # Use validation set if provided, otherwise use test set for validation
        if val_data_set:
            x_val = val_data_set.directions
            y_val = val_data_set.labels
        else:
            x_val = x_test
            y_val = y_test

    # Convert numpy arrays to PyTorch tensors if needed
    if convert_data_to_torch:
        x_train = torch.tensor(x_train)
        y_train = torch.tensor(y_train)
        x_val = torch.tensor(x_val)
        y_val = torch.tensor(y_val)
        x_test = torch.tensor(x_test)
        y_test = torch.tensor(y_test)
        
        if multi_task:
            actual_websites_train = torch.tensor(actual_websites_train)
            actual_websites_val = torch.tensor(actual_websites_val)
            actual_websites_test = torch.tensor(actual_websites_test)
    
    # Apply input preprocessing if processor is provided
    if input_processor is not None and use_input_processor:
        train_x, train_y = input_processor(x_train, y_train)
        
        if x_test is not None and len(x_test) > 0:
            x_test, y_test = input_processor(x_test, y_test)
        if x_val is not None and len(x_val) > 0:
            x_val, y_val = input_processor(x_val, y_val)
    else:
        train_x = x_train
        train_y = y_train
    
    # Clone training data for evaluation if we need to report training accuracy
    if report_train_accuracy:
        train_x_evaluation = train_x.clone()
        train_y_evaluation = train_y.clone()

    # ========================================================================
    # MULTI-TASK LEARNING SETUP
    # ========================================================================
    num_classes2 = None  # Number of classes for secondary task
    if multi_task:
        num_classes2 = cm.MON_SITE_NUM
        
        # Process secondary task labels
        if use_input_processor:
            _, train_y_wf = input_processor(None, actual_websites_train)
        else:
            train_y_wf = actual_websites_train

    # ========================================================================
    # MODEL INITIALIZATION
    # ========================================================================
    # Initialize model with default architecture if not provided
    if wf_model is None and wf_model_type is not None:
        logger.info(f'{wf_model_type} model will be initialized with the default architecture')
        
        if wf_model_type == 'RF':
            wf_model = getRF(num_classes, num2=num_classes2)
            
        elif wf_model_type == 'WFT':
            # Universal Transformer architecture for website fingerprinting
            wf_model = UTransformer(
                num_vocab=3, 
                embedding_size=128, 
                hidden_size=1024, 
                num_layers=1,
                num_heads=1, 
                total_key_depth=512, 
                total_value_depth=512,
                filter_size=512,
                classes=num_classes,
                lens=cm.trace_length,
                input_dropout=0.0,
                layer_dropout=0.0, 
                attention_dropout=0.1,
                relu_dropout=0.1,
                num_classes2=num_classes2
            )
            
        elif wf_model_type == 'tt':
            # Convolutional network architecture
            wf_model = ConvNet(
                num_classes=num_classes,
                input_shape=(1, cm.trace_length),  # (channels, sequence_length)
            )

    logger.info(f'WF model will be trained on {num_classes} classes')
    if wf_model_type is not None:
        logger.info(f'wf model type is {wf_model_type}')

    if multi_task:
        logger.info(f'{wf_model_type if wf_model_type is not None else "WF model"} is Performing Multi-Task Learning')
        mt_weight = hyperparam_manager.get('train.mt_weight')
        if mt_weight:
            logger.info(f'Multi Task learning will be performed with {mt_weight} weight')

    # ========================================================================
    # DEVICE SETUP (CPU/GPU)
    # ========================================================================
    logger.info(f'if use gpu: {if_use_gpu}')
    logger.info(f'if cuda is available: {torch.cuda.is_available()}')
    
    if cuda_id is None and if_use_gpu:
        device = torch.device('cuda' if if_use_gpu and torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(f'cuda:{cuda_id}' if if_use_gpu and torch.cuda.is_available() else 'cpu')
    
    print('device : ', device)
    wf_model.to(device)

    # ========================================================================
    # OPTIMIZER SETUP
    # ========================================================================
    if optimization_method == 'adam':
        optimizer = torch.optim.Adam(wf_model.parameters(), lr=learning_rate, weight_decay=0.001)
    
    elif optimization_method == 'step_opt':
        # Custom step optimizer (for WFT transformer)
        optimizer = StepOpt(
            len(train_x), batch_size,
            torch.optim.AdamW(wf_model.parameters(), lr=0, betas=(0.9, 0.999), 
                            weight_decay=0, amsgrad=True)
        )
    
    elif optimization_method == 'adamax':
        # Adamax optimizer (for TikTok model)
        optimizer = optim.Adamax(
            wf_model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    else:
        # Default: SGD with momentum
        optimizer = optim.SGD(wf_model.parameters(), lr=learning_rate, momentum=0.9)

    # ========================================================================
    # LEARNING RATE SCHEDULER SETUP
    # ========================================================================
    scheduler = None
    if scheduler_method == 'steplr':
        # Step decay: multiply LR by gamma every step_size epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    elif scheduler_method == 'reducelronplateau':
        # Reduce LR when validation accuracy plateaus
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10
        )

    # ========================================================================
    # DATA LOADER SETUP
    # ========================================================================
    if not multi_task:
        train_data = Data.TensorDataset(train_x, train_y)
    else:
        # Multi-task dataset with two sets of labels
        train_data = MultitaskDataset(x=train_x, y1=train_y, y2=train_y_wf)
    
    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

    # ========================================================================
    # TRAINING TRACKING VARIABLES
    # ========================================================================
    val_accuracy_list = []  # Validation accuracy per epoch
    val_accuracy_list_topks = None  # Top-k validation accuracies per epoch
    if topk_s is not None:
        val_accuracy_list_topks = [[] for k in topk_s]
    
    training_losses = []  # Training loss per epoch
    training_accuracies = []  # Training accuracy per epoch
    total_epoch_time = 0  # Cumulative time across all epochs

    # ========================================================================
    # MAIN TRAINING LOOP
    # ========================================================================
    logger.info(f'The model is being trained with {next(wf_model.parameters()).device}')
    start_time = time.time()
    logger.info('starting training')
    
    for epoch in range(total_epochs):
        epoch_start_time = time.time()
        print(f'Epoch {epoch} :')

        # Execute one training epoch
        training_stats = training_loop(
            wf_model=wf_model, 
            device=device, 
            scheduler_method=scheduler_method,
            learning_rate=learning_rate,
            train_loader=train_loader, 
            current_epoch=epoch, 
            total_epochs=total_epochs,
            if_use_gpu=if_use_gpu and torch.cuda.is_available(),
            optimizer=optimizer,
            multi_task=multi_task,
            hyperparam_manager=hyperparam_manager,
            training_loop_tqdm=training_loop_tqdm,
            class_weights=class_weights
        )
        
        # Calculate and store average epoch loss
        batch_losses = training_stats['batch_losses']
        epoch_loss = sum(batch_losses) / len(batch_losses)
        training_losses.append(epoch_loss)

        # ====================================================================
        # VALIDATION EVALUATION
        # ====================================================================
        val_accuracy = None
        val_accuracy_wf = None
        
        if should_evaluate:
            evaluation_results_val = evaluate_model(
                model=wf_model, 
                logger=logger, 
                test_x=x_val, 
                test_y=y_val, 
                batch_size=batch_size,
                top_ks=topk_s, 
                class_rankings=class_rankings_val,
                to_cpu=test_on_cpu,
                device=device,
                multi_task=multi_task,
                test_y_task2=actual_websites_val
            )
            
            val_accuracy = evaluation_results_val['total_accuracy']
            val_accuracy_wf = evaluation_results_val['total_accuracy_second_task']
            val_top_k_accuracies = evaluation_results_val['accuracies_top_k']
            
            val_accuracy_list.append(val_accuracy)
            
            # Log top-k accuracies
            if topk_s is not None:
                for idx, top_k in enumerate(topk_s):
                    topk_val = val_top_k_accuracies[idx]
                    val_accuracy_list_topks[idx].append(topk_val)
                    logger.info(f'Top {top_k} val accuracy: {topk_val:.2f}')
            
            # Log to Weights & Biases
            if use_wandb:
                wandb.log({"loss": epoch_loss, "val accuracy": val_accuracy})
                if topk_s is not None:
                    for idx, k in enumerate(topk_s):
                        wandb.log({f'val accuracy top {k}': val_top_k_accuracies[idx]})

        # ====================================================================
        # TRAINING ACCURACY EVALUATION (OPTIONAL)
        # ====================================================================
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        train_accuracy = training_stats.get('train_accuracy', None)
        train_accuracy_wf = None

        if train_accuracy is None and report_train_accuracy:
            evaluation_results_train = evaluate_model(
                model=wf_model, 
                logger=logger, 
                test_x=train_x_evaluation,
                test_y=train_y_evaluation, 
                batch_size=batch_size, 
                evaluation_mode='train',
                to_cpu=test_on_cpu,
                device=device,
                multi_task=multi_task,
                test_y_task2=actual_websites_train
            )
            train_accuracy = evaluation_results_train['total_accuracy']
            training_accuracies.append(train_accuracy)
            train_accuracy_wf = evaluation_results_train['total_accuracy_second_task']

        # Log epoch summary
        logger.info(
            f'Epoch: {epoch + 1}, epoch loss: {epoch_loss}'
            f'{f", train accuracy: {train_accuracy}" if train_accuracy else ""}'
            f'{"" if not should_evaluate else f", val accuracy: {val_accuracy}"}'
            f'{f", train accuracy wf: {train_accuracy_wf}" if train_accuracy_wf else ""}'
            f'{f", val accuracy wf: {val_accuracy_wf}" if val_accuracy_wf is not None else ""}'
        )
        
        total_epoch_time += epoch_duration
        hours, rem = divmod(epoch_duration, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info("Epoch completed in {:0>2}:{:0>2}:{:05.2f}".format(
            int(hours), int(minutes), seconds))

        # ====================================================================
        # EARLY STOPPING CHECK
        # ====================================================================
        if should_evaluate and val_acc_stop_threshold is not None:
            if val_accuracy > val_acc_stop_threshold:
                logger.info(
                    f'Current val acc ({val_accuracy:.2f}) is greater than '
                    f'{val_acc_stop_threshold:.2f}, so training will stop'
                )
                break
            else:
                logger.info(
                    f'Current val acc ({val_accuracy:.2f}) is smaller than '
                    f'{val_acc_stop_threshold:.2f}, so training will continue'
                )

        # ====================================================================
        # LEARNING RATE SCHEDULER STEP
        # ====================================================================
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_accuracy)
            else:
                scheduler.step()

    # ========================================================================
    # POST-TRAINING SUMMARY
    # ========================================================================
    average_epoch_time = total_epoch_time / (epoch + 1)
    total_time = time.time() - start_time
    hours, rem = divmod(total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    logger.info("Training completed in {:0>2}:{:0>2}:{:05.2f}".format(
        int(hours), int(minutes), seconds))

    # ========================================================================
    # FINAL TEST EVALUATION
    # ========================================================================
    if should_evaluate:
        evaluation_results_test = evaluate_model(
            model=wf_model, 
            logger=logger, 
            test_x=x_test, 
            test_y=y_test,
            batch_size=batch_size, 
            log_confusion_matrix_wandb=use_wandb,
            return_confusion_matrix=True,
            top_ks=topk_s, 
            class_rankings=class_rankings_test,
            to_cpu=test_on_cpu,
            device=device,
            evaluation_mode='test',
            actual_websites=actual_websites_test
        )
        
        test_accuracy = evaluation_results_test['total_accuracy']
        top_k_test_accuracies = evaluation_results_test['accuracies_top_k']
        confusion_matrix = evaluation_results_test['conf_matrix']
        class_accuracies = evaluation_results_test['class_accuracies']
        website_accuracies = evaluation_results_test['website_accuracies']
        
        logger.info(f'final test accuracy: {test_accuracy}')
        
        test_accuracy_top_ks = []
        if topk_s is not None:
            for idx, top_k in enumerate(topk_s):
                top_k_test = top_k_test_accuracies[idx]
                test_accuracy_top_ks.append(top_k_test)
                logger.info(f'Top {top_k} test accuracy: {top_k_test:.2f}')
        
        # Log to Weights & Biases
        if use_wandb:
            wandb.log({'test/test_accuracy': test_accuracy})
            if topk_s is not None:
                for idx, k in enumerate(topk_s):
                    wandb.log({f'test/test accuracy top {k}': test_accuracy_top_ks[idx]})

    # ========================================================================
    # MODEL SAVING
    # ========================================================================
    if save_model:
        if model_save_path is None:
            logger.info('You want the model to be saved, but have not specified its path')
        else:
            save_file(
                dir_path=model_save_path, 
                file_name='model.pth', 
                content=wf_model.state_dict()
            )

    # ========================================================================
    # PREPARE RESULTS DICTIONARY
    # ========================================================================
    training_results = {}

    if should_evaluate:
        training_results['val_accuracy_list'] = val_accuracy_list
        training_results['val_accuracy_list_topks'] = val_accuracy_list_topks
        training_results['test_accuracy'] = test_accuracy
        training_results['test_accuracy_top_ks'] = test_accuracy_top_ks
        training_results['confusion_matrix'] = confusion_matrix
        training_results['class_accuracies'] = class_accuracies
        training_results['website_accuracies'] = website_accuracies
        
        # Measure inference time
        if x_test is not None and len(x_test) > 0:
            training_results['inference_time'] = measure_inference_time(
                model=wf_model, 
                test_x=x_test,
                test_y=y_test,
                device=device,
                batch_size=batch_size
            )
        elif x_val is not None and len(x_val) > 0:
            training_results['inference_time'] = measure_inference_time(
                model=wf_model, 
                test_x=x_val,
                test_y=y_val,
                device=device,
                batch_size=batch_size
            )
    
    training_results['training_losses'] = training_losses
    training_results['training_accuracies'] = None if len(training_accuracies) == 0 else training_accuracies
    training_results['model_size'] = predict_model_size(model=wf_model)
    training_results['total_training_time'] = total_time
    training_results['average_epoch_time'] = average_epoch_time

    # Include trained model if requested
    if return_model:
        wf_model.eval()
        training_results['model'] = wf_model
    
    return training_results


### the next functions is for splitting data to train, val, test
    

def stratified_split(traces, labels, train_ratio=0.64, val_ratio=0.16, test_ratio=0.20, random_state=50, return_numpy=True, verbose=True):
    """
    Perform a stratified split of the data into train, validation, and test sets.
    Handles cases where traces have variable lengths.
    
    Parameters:
    - traces: sequence of shape (n_samples, ...) containing the input data
    - labels: sequence of shape (n_samples,) containing the labels
    - train_ratio: proportion of the data to use for training (default: 0.64)
    - val_ratio: proportion of the data to use for validation (default: 0.16)
    - test_ratio: proportion of the data to use for testing (default: 0.20)
    - random_state: random seed for reproducibility (default: 50)
    - return_numpy: if True, return NumPy arrays; if False, return original data type (default: True)
    - verbose: if True, print split information (default: True)
    
    Returns:
    - Tuple containing split datasets and indices. Returns None for skipped splits.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # Ensure ratios sum to 1
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"
    
    # Convert inputs to numpy arrays with dtype=object to handle variable-length traces
    try:
        # First try to convert to a standard numpy array (for uniform-length traces)
        traces_np = np.array(traces)
    except ValueError:
        # If that fails, use dtype=object to handle variable-length traces
        traces_np = np.array(traces, dtype=object)
    
    labels_np = np.array(labels)
    
    # Total number of samples
    total_count = len(labels_np)
    
    # Initialize return variables
    X_val_np = y_val_np = X_test_np = y_test_np = None
    val_indices = test_indices = None
    
    # Case 1: Only training data (no validation or test)
    if val_ratio == 0 and test_ratio == 0:
        X_train_np, y_train_np = traces_np, labels_np
        train_index = np.arange(total_count)
    
    # Case 2: Training and validation data (no test)
    elif test_ratio == 0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=random_state)
        for train_index, val_indices in sss.split(traces_np, labels_np):
            X_train_np, y_train_np = traces_np[train_index], labels_np[train_index]
            X_val_np, y_val_np = traces_np[val_indices], labels_np[val_indices]
    
    # Case 3: Training and test data (no validation)
    elif val_ratio == 0:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
        for train_index, test_indices in sss.split(traces_np, labels_np):
            X_train_np, y_train_np = traces_np[train_index], labels_np[train_index]
            X_test_np, y_test_np = traces_np[test_indices], labels_np[test_indices]
    
    # Case 4: All three splits (original behavior)
    else:
        # First split: separate train from temp (val + test combined)
        sss_train = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=random_state)
        
        for train_index, temp_index in sss_train.split(traces_np, labels_np):
            X_train_np, y_train_np = traces_np[train_index], labels_np[train_index]
            X_temp, y_temp = traces_np[temp_index], labels_np[temp_index]
        
        # Second split: separate val and test from temp
        sss_val_test = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=test_ratio / (val_ratio + test_ratio), 
            random_state=random_state
        )
        
        for val_index_temp, test_index_temp in sss_val_test.split(X_temp, y_temp):
            X_val_np, y_val_np = X_temp[val_index_temp], y_temp[val_index_temp]
            X_test_np, y_test_np = X_temp[test_index_temp], y_temp[test_index_temp]
            
            # Calculate the actual indices in the original dataset
            val_indices = temp_index[val_index_temp]
            test_indices = temp_index[test_index_temp]
    
    # Print summary of the split
    if verbose:
        print(f"Total samples: {total_count}")
        print(f"Train set: {len(X_train_np)} samples ({len(X_train_np)/total_count:.2%})")
        
        if X_val_np is not None:
            print(f"Validation set: {len(X_val_np)} samples ({len(X_val_np)/total_count:.2%})")
        else:
            print("Validation set: None")
            
        if X_test_np is not None:
            print(f"Test set: {len(X_test_np)} samples ({len(X_test_np)/total_count:.2%})")
        else:
            print("Test set: None")
    
        # Print class distribution in each set
        sets_to_print = [("Train", y_train_np)]
        if y_val_np is not None:
            sets_to_print.append(("Validation", y_val_np))
        if y_test_np is not None:
            sets_to_print.append(("Test", y_test_np))
            
        for name, y in sets_to_print:
            unique, counts = np.unique(y, return_counts=True)
            print(f"\n{name} set class distribution:")
            for class_label, count in zip(unique, counts):
                print(f"Class {class_label}: {count} samples ({count/len(y):.2%})")
    
    if return_numpy:
        return_values = [X_train_np, y_train_np, X_val_np, y_val_np, X_test_np, y_test_np, train_index, val_indices, test_indices]
        return_values = [np.array([]) if x is None else x for x in return_values]
        return tuple(return_values)
    else:
        # Convert back to original data type
        X_train = type(traces)(X_train_np)
        y_train = type(labels)(y_train_np)
        X_val = X_val_np if X_val_np is None else type(traces)(X_val_np)
        y_val = y_val_np if y_val_np is None else type(labels)(y_val_np)
        X_test = X_test_np if X_test_np is None else type(traces)(X_test_np)
        y_test = y_test_np if y_test_np is None else type(labels)(y_test_np)

        # Convert any None values to empty lists
        return_values = [X_train, y_train, X_val, y_val, X_test, y_test, train_index, val_indices, test_indices]
        return_values = [[] if x is None else x for x in return_values]
        return tuple(return_values)

# Example usage:
# X_train, y_train, X_val, y_val, X_test, y_test, train_indices, val_indices, test_indices = stratified_split(ordered_traces, ordered_labels, return_numpy=True)

# If you need torch tensors:
# X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
# X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
# X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)

# Example usage:
# X_train, y_train, X_val, y_val, X_test, y_test, train_indices, val_indices, test_indices = stratified_split(ordered_traces, ordered_labels)

# If you need torch tensors:
# X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
# X_val, y_val = torch.tensor(X_val), torch.tensor(y_val)
# X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)
import numpy as np
import matplotlib.pyplot as plt

def show_class_distribution(ordered_labels, train_indices, val_indices, plot = False):
    """
    Prints and plots the distribution (percentages) of each class in train and val splits.
    
    Parameters
    ----------
    ordered_labels : array-like
        The list/array of class (cluster) labels for the entire dataset.
    train_indices : array-like
        Indices of the training subset.
    val_indices : array-like
        Indices of the validation subset.
    """

    # Convert to numpy arrays if not already
    ordered_labels = np.array(ordered_labels)
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    # Get the labels for training and validation subsets
    train_labels = ordered_labels[train_indices]
    val_labels = ordered_labels[val_indices]
    
    # Identify unique classes
    unique_labels = np.unique(ordered_labels)

    # Prepare a dict to store percentage distribution
    class_percentages = {}

    for label in unique_labels:
        total_count = np.sum(ordered_labels == label)
        train_count = np.sum(train_labels == label)
        val_count = np.sum(val_labels == label)

        # Calculate the percentage of this class in train and val
        train_percent = (train_count / total_count) * 100 if total_count > 0 else 0
        val_percent = (val_count / total_count) * 100 if total_count > 0 else 0

        class_percentages[label] = (train_percent, val_percent)

    # Print results
    print("Class distribution (percentage) in train and val sets:")
    for label in unique_labels:
        train_percent, val_percent = class_percentages[label]
        print(f"Class {label}: Train = {train_percent:.2f}%, Val = {val_percent:.2f}%")

    # Plot results
    if plot:
        x = np.arange(len(unique_labels))
        train_values = [class_percentages[label][0] for label in unique_labels]
        val_values   = [class_percentages[label][1] for label in unique_labels]

        bar_width = 0.35
        fig, ax = plt.subplots()
        ax.bar(x - bar_width/2, train_values, bar_width, label='Train')
        ax.bar(x + bar_width/2, val_values, bar_width, label='Val')

        ax.set_xlabel('Class Label')
        ax.set_ylabel('Percentage')
        ax.set_title('Class Distribution in Train vs. Val')
        ax.set_xticks(x)
        ax.set_xticklabels(unique_labels)
        ax.legend()

        plt.tight_layout()
        plt.show()



def per_class_accuracy(confusion_matrix):
    # Diagonal elements are the correctly classified samples for each class
    class_correct = np.diag(confusion_matrix)
    
    # Sum of each row gives the total number of samples for each class
    class_total = np.sum(confusion_matrix, axis=1)
    
    # Per-class accuracy
    per_class_acc = class_correct / class_total
    
    return per_class_acc







def plot_per_class_accuracy(per_class_acc, save_path, class_names=None):
    # Create class labels if not provided
    if class_names is None:
        class_names = [f'Class {i}' for i in range(len(per_class_acc))]
    
    # Ensure per_class_acc and class_names have the same length
    if len(per_class_acc) != len(class_names):
        raise ValueError("The number of accuracies must match the number of class names.")

    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, per_class_acc)
    plt.title('Per-Class Accuracy')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)  # Set y-axis limit from 0 to 1

    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    # Rotate x-axis labels if there are many classes
    if len(class_names) > 10:
        plt.xticks(rotation=45, ha='right')

    save_file(dir_path=save_path, file_name=f'per_class_accuracies.png')

    

# Example usage:
# per_class_acc = [0.85, 0.92, 0.78]  # Example accuracies
# plot_per_class_accuracy(per_class_acc, class_names=['Cat', 'Dog', 'Bird'])


def format_time(seconds: float) -> str:
    """
    Convert seconds to a human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        str: Formatted time string (e.g., "2h 30m 45s" or "45m 30s" or "30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:  # include seconds if it's the only non-zero value
        parts.append(f"{secs}s")
    
    return " ".join(parts)



def plot_model_stats(
    train_results: Dict[str, Any], 
    title: str = "Model Statistics", 
    save_path: str = None,
    save_format: str = 'png',
    dpi: int = 300,
    **kwargs
) -> None:
    """
    Create and optionally save a visual summary of model statistics.
    
    Args:
        train_results: Dictionary containing training results
        title: Plot title
        save_path: Path where to save the figure. If None, figure is not saved
        save_format: Format to save the figure ('png', 'pdf', 'svg', etc.)
        dpi: Resolution for saving the figure
        **kwargs: Additional key-value pairs to display on the plot
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    # Extract core metrics
    metrics = {
        'Final Training Loss': f"{train_results['training_losses'][-1]:.2f}",
        'Final Training Accuracy': f"{train_results['training_accuracies'][-1]*100:.2f}%",
        'Final Validation Accuracy': f"{train_results['val_accuracy_list'][-1]*100:.2f}%",
        'Test Accuracy': f"{train_results['test_accuracy']*100:.2f}%",
        'Model Size': f"{train_results['model_size']:.4f} GB",
        'Inference Time': f"{train_results['inference_time']*1000:.2f} ms",
        'Total Training Time': format_time(train_results['total_training_time']),
        'Average Epoch Time': format_time(train_results['average_epoch_time'] )
    }
    
    # Add additional metrics from kwargs
    for key, value in kwargs.items():
        # Format numbers to 2 decimal places if they're floats
        if isinstance(value, float):
            metrics[key] = f"{value:.2f}"
        else:
            metrics[key] = str(value)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title, pad=20, size=14, fontweight='bold')
    
    # Remove axes
    ax.set_axis_off()
    
    # Calculate grid dimensions
    n_metrics = len(metrics)
    n_cols = 3  # We'll use 3 columns
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    # Calculate positions for text
    cell_height = 1 / n_rows
    cell_width = 1 / n_cols
    padding = 0.1  # Padding between cells
    
    # Create background rectangles and text
    for idx, (key, value) in enumerate(metrics.items()):
        # Calculate position
        row = idx // n_cols
        col = idx % n_cols
        
        # Calculate box position
        box_x = col * cell_width + padding/2
        box_y = 1 - (row + 1) * cell_height + padding/2
        box_width = cell_width - padding
        box_height = cell_height - padding
        
        # Create box
        rect = plt.Rectangle(
            (box_x, box_y), 
            box_width, 
            box_height, 
            facecolor='lightgray' if idx % 2 == 0 else 'white',
            alpha=0.3,
            transform=ax.transAxes,
            edgecolor='gray',
            linewidth=1
        )
        ax.add_patch(rect)
        
        # Add text
        ax.text(
            box_x + box_width/2,
            box_y + box_height*0.6,
            key,
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontweight='bold',
            fontsize=10
        )
        ax.text(
            box_x + box_width/2,
            box_y + box_height*0.3,
            value,
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=12,
            color='darkblue'
        )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path is not None:
        save_file(dir_path= save_path, file_name = 'training_stats.png')
        

# Example usage:
"""
# Save as PNG
fig = plot_model_stats(
    train_results,
    "Model Performance Summary",
    save_path="results/model_stats.png",
    Learning_Rate=0.001,
    Batch_Size=32
)

# Save as PDF
fig = plot_model_stats(
    train_results,
    "Model Performance Summary",
    save_path="results/model_stats.pdf",
    save_format='pdf',
    Learning_Rate=0.001,
    Batch_Size=32
)

plt.show()  # Display the plot
plt.close()  # Close the figure to free memory
"""




def plot_website_accuracies(save_path, website_accuracies = None, figsize=(12, 6), title="Website Accuracies"):
    """
    Plot accuracies for different websites using a bar plot, highlighting the 5 lowest performing websites
    
    Parameters:
    website_accuracies (dict): Dictionary with website numbers as keys and accuracies as values
    figsize (tuple): Figure size as (width, height)
    title (str): Plot title
    """
    if website_accuracies is None:
        return
        
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get websites and accuracies
    websites = list(website_accuracies.keys())
    accuracies = list(website_accuracies.values())
    
    # Find indices of 5 lowest accuracies
    lowest_indices = sorted(range(len(accuracies)), key=lambda i: accuracies[i])[:5]
    lowest_websites = [websites[i] for i in lowest_indices]
    lowest_accuracies = [accuracies[i] for i in lowest_indices]
    
    # Create bar plot
    bars = ax.bar(websites, accuracies, color='skyblue', alpha=0.7)
    
    # Highlight the 5 lowest accuracy bars in different shades of red
    red_colors = ['#ff0000', '#ff3333', '#ff6666', '#ff9999', '#ffcccc']
    for idx, (low_idx, color) in enumerate(zip(lowest_indices, red_colors)):
        bars[low_idx].set_color(color)
        bars[low_idx].set_alpha(0.7)
        
        # Add text annotation for each lowest accuracy
        ax.annotate(f'#{idx+1} Lowest: {accuracies[low_idx]:.3f}',
                    xy=(websites[low_idx], accuracies[low_idx]),
                    xytext=(10, 10 + idx * 20),  # Stagger annotations vertically
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Customize plot
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('Website Number', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels if there are many websites
    if len(websites) > 20:
        plt.xticks(rotation=45)
    
    # Add mean line
    mean_accuracy = sum(accuracies) / len(accuracies)
    ax.axhline(y=mean_accuracy, color='green', linestyle='--', alpha=0.5, 
               label=f'Mean Accuracy: {mean_accuracy:.3f}')
    
    # Add legend entries
    for idx, (website, color) in enumerate(zip(lowest_websites, red_colors)):
        ax.bar([], [], color=color, alpha=0.7, 
               label=f'#{idx+1} Lowest (Website {website})')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    save_file(dir_path=save_path, file_name='per_website_accuracy.png')




def remap_labels(labels):
    # some times, the labels in a cluster are like [12, 23, 35]. I want to make it 0,1,2
    # Get unique values and create a mapping dictionary
    unique_labels = list(set(labels))
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    
    # Convert to numpy array if it isn't already
    labels = np.array(labels)
    
    # Create new array with mapped values
    remapped_labels = np.array([label_map[label] for label in labels])
    
    return remapped_labels, label_map

def finetune_pretrained_model_on_single_cluster(traces, labels, use_gpu, logger, training_config, model_type = 'RF', should_evaluate = True, training_loop_tqdm = True,
                                                report_train_accuracy = True, return_validation_data = False ):
    # finetuning a given model on a cluster (defended or undefended). useful in regulator optimization. currently only supproting RF
    number_of_unique_websites = len(set(labels)) # TODO the classes might be imbalanced. add this in the loss function?
    logger.info(f'Finetuning {model_type} model on cluster with {len(traces)} instances and {number_of_unique_websites} websites')
    pretrained_wf_model = None
    labels, _ = remap_labels(labels)
    if model_type == 'RF': # we expect the traces to be tams
        pretrained_save_path = os.path.join(cm.BASE_DIR, 'models', 'RF_Original', f'{cm.data_set_folder}', 'model.pth')
        original_cfg = {
        'N': [128, 128, 'M', 256, 256, 'M', 512]
            } # TODO, don't hard code this
        pretrained_wf_model = load_and_modify_RF_model(pretrained_path= pretrained_save_path,
                                    original_cfg= original_cfg,
                                    original_num_classes= cm.MON_SITE_NUM,
                                    new_num_classes= number_of_unique_websites, 
                                    freeze_features= True)
        
        
        
        training_loop = rf_training_loop
        input_processor = RF_input_processor
    else:
        pass # TODO

    
    X_train, y_train, X_val, y_val, X_test, y_test, train_indices, val_indices, test_indices = stratified_split(traces, labels,
                                                                                                                train_ratio= 0.8,
                                                                                                                val_ratio= 0.2,
                                                                                                                test_ratio= 0)
    
    
    
    if use_gpu and torch.cuda.is_available():
        logger.info('Since you requested for gpu and gpu is availabel, gpu will be used')
        if_use_gpu = 1
    else:
        if_use_gpu = 0
        logger.info('GPU is not available, thus training will be done on CPU')
    
    training_config_dir = os.path.join(cm.BASE_DIR, 'configs', 'training', training_config + '.yaml')
    hyperparam_manager = ConfigManager(config_path= training_config_dir)

    # we want the retraining to be as fast as possible. we will not evaluate the model during each epoch and do it only once at the end
    training_results = train_wf_model(  wf_model= pretrained_wf_model,
                                           logger= logger, 
                    training_loop= training_loop  ,                                                                                                          
                    num_classes= cm.MON_SITE_NUM,
                    input_processor= input_processor,
                    hyperparam_manager= hyperparam_manager,
                    save_model= False,
                    x_train = X_train,
                    y_train = y_train,
                    x_test= X_test,
                    y_test= y_test,
                    x_val= X_val,
                    y_val= y_val,
                    if_use_gpu= if_use_gpu,
                    report_train_accuracy= report_train_accuracy,
                    wf_model_type= model_type,
                    should_evaluate= should_evaluate,
                    return_model= True,
                    training_loop_tqdm= training_loop_tqdm)

    if not return_validation_data:
        return training_results
    else:
        return training_results, X_val, y_val
