import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from .train_utils import tt_input_processor
class ConvNet(nn.Module):
    def __init__(self, 
                 num_classes,
                 input_shape,
                 activation_functions=("elu", "relu", "relu", "relu", "relu", "relu"),
                 dropout_rates=(0.1, 0.1, 0.1, 0.1, 0.5, 0.7),
                 filter_nums=(32, 64, 128, 256),
                 kernel_size=8,
                 conv_stride_size=1,
                 pool_stride_size=4,
                 pool_size=8,
                 fc_layer_sizes=(512, 512)):
        super(ConvNet, self).__init__()
        
        # Input shape should be (channels, sequence_length)
        self.input_channels = input_shape[0]
        
        # Validation
        assert len(filter_nums) + len(fc_layer_sizes) <= len(activation_functions)
        assert len(filter_nums) + len(fc_layer_sizes) <= len(dropout_rates)
        
        # Create ModuleList to hold convolutional blocks
        self.conv_blocks = nn.ModuleList()
        
        # Function to get activation
        def get_activation(name):
            return nn.ELU() if name == 'elu' else nn.ReLU()
        
        # Build convolutional blocks
        in_channels = self.input_channels
        for i, filters in enumerate(filter_nums):
            conv_block = nn.Sequential(
                # First Conv1D
                nn.Conv1d(in_channels, filters, kernel_size, 
                         stride=conv_stride_size, padding='same'),
                nn.BatchNorm1d(filters),
                get_activation(activation_functions[i]),
                
                # Second Conv1D
                nn.Conv1d(filters, filters, kernel_size,
                         stride=conv_stride_size, padding='same'),
                nn.BatchNorm1d(filters),
                get_activation(activation_functions[i]),
                
                # Pooling and Dropout
                nn.MaxPool1d(pool_size, stride=pool_stride_size, padding=pool_size//2),
                nn.Dropout(dropout_rates[i])
            )
            self.conv_blocks.append(conv_block)
            in_channels = filters
        
        # Calculate the output size of the last conv layer
        self.feature_size = self._get_conv_output_size(input_shape)
        
        # Create ModuleList for fully connected layers
        self.fc_blocks = nn.ModuleList()
        
        # Build fully connected blocks
        in_features = self.feature_size
        for i, fc_size in enumerate(fc_layer_sizes):
            fc_block = nn.Sequential(
                nn.Linear(in_features, fc_size),
                nn.BatchNorm1d(fc_size),
                get_activation(activation_functions[len(filter_nums) + i]),
                nn.Dropout(dropout_rates[len(filter_nums) + i])
            )
            self.fc_blocks.append(fc_block)
            in_features = fc_size
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes),
            nn.Softmax(dim=1)
        )
    
    def _get_conv_output_size(self, shape):
        # Helper function to calculate conv output size
        batch_size = 1
        input = torch.autograd.Variable(torch.rand(batch_size, *shape))
        output = input
        
        for conv_block in self.conv_blocks:
            output = conv_block(output)
        
        return int(torch.prod(torch.tensor(output.size()[1:])))
    
    def forward(self, x):
        # Apply conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Apply fc blocks
        for fc_block in self.fc_blocks:
            x = fc_block(x)
        
        # Apply final classifier
        x = self.classifier(x)
        
        return x
    
if __name__ == '__main__':
    # Create sample batch data
    batch_size = 32
    input_length = 5000
    batch_data = np.random.randn(batch_size, input_length)  # Create 32 random samples

    # Create model instance
    model = ConvNet(
        num_classes=10,
        input_shape=(1, input_length),  # (channels, sequence_length)
        activation_functions=("elu", "relu", "relu", "relu", "relu", "relu"),
        dropout_rates=(0.1, 0.1, 0.1, 0.1, 0.5, 0.7),
        filter_nums=(32, 64, 128, 256),
        fc_layer_sizes=(512, 512)
    )

    # Set model to evaluation mode
    model.eval()

    # Prepare input tensor
    # Shape will be: (batch_size, channels, sequence_length)
    # input_tensor = torch.from_numpy(batch_data).float()
    # input_tensor = input_tensor.unsqueeze(1)  # Add channel dimension
    input_tensor, _ = tt_input_processor(x = [np.zeros([input_length])], y = None, trace_length= input_length)
    print(input_tensor.shape)
    # Perform forward pass
    with torch.no_grad():
        output = model(input_tensor)
        print(output.shape)