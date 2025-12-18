import torch
from tqdm import tqdm
import utils.config_utils as cm
import numpy as np

def tt_training_loop(wf_model, device, train_loader, optimizer, learning_rate, 
                 current_epoch, total_epochs, class_weights =None, **kwargs):
    """
    Training loop function for a single epoch
    
    Args:
        wf_model: The model to train
        device: The device to train on (cuda/cpu)
        train_loader: DataLoader containing the training data
        optimizer: The optimizer instance
        learning_rate: Learning rate for optimization
        current_epoch: Current epoch number
        total_epochs: Total number of epochs
        **kwargs: Additional arguments
        
    Returns:
        list: Batch losses for the epoch
    """
    
    # Set model to training mode
    wf_model.train()
    
    # Initialize list to store batch losses
    batch_losses = []
    
    # Create progress bar
    progress_bar = tqdm(train_loader, 
                       desc=f'Training Epoch {current_epoch}/{total_epochs}',
                       leave=True)
    loss_func = torch.nn.CrossEntropyLoss(weight= class_weights)
    # Training loop over batches
    for batch_idx, (data, target) in enumerate(progress_bar):
        # Move data to device
        data, target = data.to(device), target.to(device)
        
        # Zero gradients for this batch
        optimizer.zero_grad()
        
        # Forward pass
        output = wf_model(data)
        
        # Calculate loss
        loss = loss_func(output, target)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        # Store batch loss
        batch_losses.append(loss.item())
        
        # Update progress bar description with current loss
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{sum(batch_losses)/len(batch_losses):.4f}'
        })
    
    training_stats = {}
    training_stats['batch_losses'] = batch_losses
    return training_stats



def tt_input_processor(x, y, trace_length = None):
    """
    Process input data for the model by padding or truncating sequences and converting to PyTorch tensors.
    Returns both processed x and y tensors, even if one input is None.
    
    Args:
        x (list or None): List of numpy arrays, each of shape (d,) where d can vary
        y (list or None): List of labels
        trace_length (int): Target length for all sequences (cm.trace_length)
    
    Returns:
        tuple: (x_tensor, y_tensor) - Both can be None if their inputs were None
    """
    # Process x if it exists
    if trace_length is None:
        trace_length = cm.trace_length
    if x is not None and len(x) > 0:
        num_sequences = len(x)
        processed_x = np.zeros((num_sequences, trace_length))
        
        for i, sequence in enumerate(x):
            sequence_length = len(sequence)
            if sequence_length >= trace_length:
                processed_x[i] = sequence[:trace_length]
            else:
                processed_x[i, :sequence_length] = sequence
        
        x_tensor = torch.FloatTensor(processed_x).unsqueeze(1)
    else:
        x_tensor = None
    
    # Process y if it exists
    if y is not None and len(y) > 0:
        y_tensor = torch.LongTensor(y)
    else:
        y_tensor = None
    
    return x_tensor, y_tensor