import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from preprocessing import get_volume
from icecream import ic

class Config:
    dataset_path = os.path.expanduser('~/Desktop/UniToBrain')
    device = torch.device('mps') or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Training parameters
    batch_size = 4
    sequence_length = 4
    learning_rate = 1e-4
    num_epochs = 100
    
    # Data parameters
    train_split = 0.8
    val_split = 0.1
    # test_split will be the remainder

class PerfusionDataset(Dataset):
    def __init__(self, folder_paths, sequence_length=4, transform=None):
        self.folder_paths = folder_paths
        self.sequence_length = sequence_length
        self.transform = transform
        
    def __len__(self):
        return len(self.folder_paths)
    
    def __getitem__(self, idx):
        # Load volume sequence
        volume_seq = get_volume(
            self.folder_paths[idx],
            windowing=True,
            windowing_type='brain',
            extract_brain=True,
            correct_motion=True
        )
        
        # Convert to tensor
        volume_seq = torch.from_numpy(volume_seq).float()
        
        # Get input sequence and target
        input_seq = volume_seq[:self.sequence_length]
        target = volume_seq[self.sequence_length:self.sequence_length+1]
        
        if self.transform:
            input_seq = self.transform(input_seq)
            target = self.transform(target)
            
        return input_seq, target

def get_data_loaders(batch_size=4, sequence_length=4):
    # Load all folder paths
    folder_paths = []
    for folder in sorted(os.listdir(Config.dataset_path)):
        folder_path = os.path.join(Config.dataset_path, folder)
        if len(folder) == 7:  # MOL-XYZ format
            folder_paths.append(folder_path)
            
    # Split into train/val/test
    n_train = int(len(folder_paths) * Config.train_split)
    n_val = int(len(folder_paths) * Config.val_split)
    
    train_paths = folder_paths[:n_train]
    val_paths = folder_paths[n_train:n_train+n_val]
    test_paths = folder_paths[n_train+n_val:]
    
    # Create datasets
    train_dataset = PerfusionDataset(train_paths, sequence_length)
    val_dataset = PerfusionDataset(val_paths, sequence_length)
    test_dataset = PerfusionDataset(test_paths, sequence_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
        