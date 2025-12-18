import torch.nn as nn
import torch.nn as nn

class MultitaskLoss:
    def __init__(self, alpha=0.7, criterion=None):
        self.alpha = alpha  # Weight for y1 task (primary task)
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion

    def __call__(self, y1_pred, y2_pred, y1_true, y2_true):
        # Calculate individual losses
        loss_y1 = self.criterion(y1_pred, y1_true)
        loss_y2 = self.criterion(y2_pred, y2_true)

        
        # Combine losses while maintaining computational graph
        total_loss = self.alpha * loss_y1 + (1 - self.alpha) * loss_y2 
        breakpoint()
        #total_loss = loss_y1 + loss_y2
        return total_loss
