import torch
from torch.utils.data import Dataset, DataLoader
class MultitaskDataset(Dataset):
    def __init__(self, x, y1, y2):
        """
        Args:
            x: Input features
            y1: First set of labels
            y2: Second set of labels
        """
        self.x = x
        self.y1 = y1
        self.y2 = y2
        # print(f'y1 is shape {y1.shape}')
        # print(f'y2 is shape {y2.shape}')
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y1[idx], self.y2[idx]