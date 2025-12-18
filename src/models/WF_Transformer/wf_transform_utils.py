# A set of functions regarding the wf transformer model
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional, Union, Dict
from .UTransformer import UTransformer, ClassifierModule
def load_pretrained_transformer(
    pretrained_model_path: str,
    new_num_classes: Optional[int] = None,
    freeze_config: Dict[str, bool] = {
        "embedding": True,
        "transformer": True,
        "compression": True,
        "classifier": False
    },
    classifier_hidden_size: Optional[int] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> UTransformer:
    """
    Load a pretrained ModularTransformer model and optionally modify its classifier head
    and freeze specific modules.
    
    Args:
        pretrained_model_path: Path to the pretrained model state dict
        new_num_classes: Number of classes for the new classifier head (if None, keep original)
        freeze_config: Dictionary specifying which modules to freeze
        classifier_hidden_size: Size of hidden layer in new classifier (if None, keep original)
        device: Device to load the model on
        
    Returns:
        Modified ModularTransformer model
    """
    # Load the pretrained model
    model = torch.load(pretrained_model_path)
    
    # If it's just the state dict, load it directly
    # if isinstance(pretrained_state, dict):
    #     model_state = pretrained_state
    #     model = UTransformer(**get_model_config_from_state_dict(model_state))
    #     model.load_state_dict(model_state)
    # else:
    #     # If it's the whole model
    #     model = pretrained_state
    
    # Move model to device
    model = model.to(device)
    
    # If we want to modify the classifier
    if new_num_classes is not None or classifier_hidden_size is not None:
        # Get current classifier config
        old_classifier = model.classifier
        current_hidden_size = old_classifier.fc1.out_features
        current_input_size = old_classifier.fc1.in_features
        
        # Create new classifier with specified or original hidden size
        new_classifier = ClassifierModule(
            input_size=current_input_size,
            hidden_size=classifier_hidden_size or current_hidden_size,
            num_classes=new_num_classes or old_classifier.fc2.out_features
        )
        
        # Replace the classifier
        model.classifier = new_classifier.to(device)
    
    # Freeze/unfreeze modules based on config
    for module_name, should_freeze in freeze_config.items():
        if hasattr(model, module_name):
            for param in getattr(model, module_name).parameters():
                param.requires_grad = not should_freeze
    
    return model


def get_model_config_from_state_dict(state_dict: Dict) -> Dict:
    """
    Extract model configuration from state dict keys.
    This is a helper function to reconstruct model architecture.
    """
    config = {}
    
    # Extract embedding size from embedding weight matrix
    embed_weight = state_dict['embedding.emb.weight']
    config['num_vocab'] = embed_weight.size(0)
    config['embedding_size'] = embed_weight.size(1)
    
    # Extract hidden size from transformer layer norm
    config['hidden_size'] = state_dict['transformer.ln.weight'].size(0)
    
    # Extract other configurations if possible
    # Note: Some of these might need to be provided explicitly if not inferrable from state dict
    config.update({
        'num_layers': 6,  # default value, might need to be adjusted
        'num_heads': 8,  # default value, might need to be adjusted
        'total_key_depth': 64,  # default value
        'total_value_depth': 64,  # default value
        'filter_size': 128,  # default value
        'lens': 512,  # default value
        'max_length': 5000,  # default value
        'input_dropout': 0.0,
        'layer_dropout': 0.0,
        'attention_dropout': 0.0,
        'relu_dropout': 0.0,
        'num_classes': state_dict['classifier.fc2.weight'].size(0)
    })
    
    return config

def WFT_input_processor(x_input =None, y_input = None):
    # we give the input to this function and turn them into tensors ready to be fed into an wft model.
    x = None
    y = None

    if x_input is not None:
        lens = x_input.shape[1]
        x = x_input.view([-1,lens,1]).long()
    if y_input is not None:
        y = y_input.view([-1]).long()

    return x, y