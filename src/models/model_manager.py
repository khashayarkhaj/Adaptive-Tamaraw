import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from RF.RF_model import getRF
class ModelManager:
    # A class for combining all the different models together
    def __init__(self, num_classes = 95, model_type = 'RF'):

        assert model_type in ['RF'], f'Model type {model_type} is not in the pre-defined models.'
        self.model_type = model_type
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.num_classes = num_classes

    def build_model(self, ):
        if self.model_type == 'RF':
            self.model = getRF(self.num_classes)
            
        # Build model based on the model type
        pass

    def train(self, train_loader, device):

        # Train the model based on the model type
        pass

    def evaluate(self, test_loader, device):
        # Evaluate the model based on the model type
        pass