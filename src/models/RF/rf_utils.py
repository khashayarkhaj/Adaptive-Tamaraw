# A set of functions regarding the rf model
import numpy as np
import torch
import torch.nn as nn
from .RF_model import getRF, make_classifier

def RF_input_processor(x_input = None, y_input = None):
    # we give the input tams to this function and turn them into tensors ready to be fed into an rf model.

    if x_input is not None:
        if not isinstance(x_input, np.ndarray):
            x_input = np.array(x_input)
        x_input = torch.unsqueeze(torch.from_numpy(x_input), dim=1).type(torch.FloatTensor)
        x_input = x_input.view(x_input.size(0), 1, 2, -1)
    if y_input is not None:
        if not isinstance(x_input, np.ndarray):
            y_input = np.array(y_input)
        y_input= torch.from_numpy(y_input).type(torch.LongTensor)

    return x_input, y_input



import torch
import torch.nn as nn

def load_and_modify_RF_model(
    pretrained_path, 
    original_cfg = None,
    original_num_classes = 95, 
    new_num_classes=None,
    freeze_features=True,
    freeze_first_layer=True,
    device = 'cuda',
):
    """
    Load a pretrained RF model with control over which sections to freeze
    
    Args:
        pretrained_path (str): Path to the pretrained .pth file
        new_num_classes (int, optional): Number of classes for the new classifier
        freeze_features (bool): Whether to freeze the features section
        freeze_first_layer (bool): Whether to freeze the first_layer section
    
    Returns:
        RF: Modified model
    """
    # Load the state dict
    if not torch.cuda.is_available():
        device = 'cpu'
    state_dict = torch.load(pretrained_path, map_location=device)
    
    # Create a new model instance
    
    model = getRF(cfg= original_cfg, num= original_num_classes)  # Use original num_classes initially  # Use original num_classes initially
    
    # Load the pretrained weights
    model.load_state_dict(state_dict)
    feature_configs = model.feature_configs
    # Freeze/unfreeze first_layer
    for param in model.first_layer.parameters():
        param.requires_grad = not freeze_first_layer
        
    # Freeze/unfreeze features
    for param in model.features.parameters():
        param.requires_grad = not freeze_features
    
    # If we want to modify the classifier
    if new_num_classes is not None:
        input_channels = feature_configs['N'][-1]
        model.main_classifier = make_classifier(
            input_channels=input_channels, 
            num_classes=new_num_classes
        )
        
    
    # Classifier is always trainable
    for param in model.main_classifier.parameters():
        param.requires_grad = True

    if model.multi_task_enabled:
        for param in model.classifier_y2.parameters():
            param.requires_grad = True

    
    return model

# Utility function to check which parts are trainable
def print_model_status(model):
    """
    Print the trainable status of each part of the model
    """
    def count_parameters(parameters):
        return sum(p.numel() for p in parameters if p.requires_grad)
    
    print("\nModel Status:")
    print("-" * 50)
    
    # Check first_layer
    first_layer_trainable = any(p.requires_grad for p in model.first_layer.parameters())
    print(f"First Layer: {'Trainable' if first_layer_trainable else 'Frozen'}")
    print(f"Trainable parameters: {count_parameters(model.first_layer.parameters()):,}")
    
    # Check features
    features_trainable = any(p.requires_grad for p in model.features.parameters())
    print(f"\nFeatures: {'Trainable' if features_trainable else 'Frozen'}")
    print(f"Trainable parameters: {count_parameters(model.features.parameters()):,}")
    
    # Check classifier
    classifier_trainable = any(p.requires_grad for p in model.classifier.parameters())
    print(f"\nClassifier: {'Trainable' if classifier_trainable else 'Frozen'}")
    print(f"Trainable parameters: {count_parameters(model.classifier.parameters()):,}")
    
    # Total trainable parameters
    total_trainable = count_parameters(model.parameters())
    print(f"\nTotal trainable parameters: {total_trainable:,}")

# Example usage:

# 1. Load model with everything frozen except classifier
# model1 = load_and_modify_model(
#     pretrained_path='path/to/model.pth',
#     new_num_classes=10,
#     freeze_features=True,
#     freeze_first_layer=True
# )

# # 2. Load model with features trainable
# model2 = load_and_modify_model(
#     pretrained_path='path/to/model.pth',
#     new_num_classes=10,
#     freeze_features=False,  # Features will be trainable
#     freeze_first_layer=True
# )

# # 3. Load model with everything trainable
# model3 = load_and_modify_model(
#     pretrained_path='path/to/model.pth',
#     new_num_classes=10,
#     freeze_features=False,
#     freeze_first_layer=False
# )

# # Check the status of your model
# print_model_status(model1)

def unfreeze_layers(model, layers_to_unfreeze):
    """
    Unfreeze specific layers in the model
    
    Args:
        model (RF): The model
        layers_to_unfreeze (list): List of layer names to unfreeze
    """
    for name, param in model.named_parameters():
        for layer_name in layers_to_unfreeze:
            if layer_name in name:
                param.requires_grad = True
                break

# Example: Unfreeze specific layers
# unfreeze_layers(model, ['features.0', 'first_layer'])


def rf_model_inference(rf_model, input_tam, label = None):
    # given an rf model and an input tam, it performs inference on the input. if label is given, it returns wether the prediction was correct
    
    rf_model.eval() ### this seems to be crucial because it shuts down dropout and batch norm
    print('eval line')
    X_t, label = RF_input_processor(x_input= [input_tam], y_input= [label])
    print('processor line')
    result = rf_model(X_t)
    print('inference line')
    if label is None:
        print('if label line')
        return result
    else:
        result = result.cpu()
        label = label.cpu()
        print('cpu line')
        pred_y = torch.max(result, 1)[1].data.numpy().squeeze()
        print('pred line')
        accuracy = (pred_y == label.numpy()).sum().item() * 1.0 / float(label.size(0))
        print('accuracy line')
        return accuracy