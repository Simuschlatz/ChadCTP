import os
import numpy as np
from torch.utils.data import Dataset
import torch
from preprocessing import DataTransformer
from icecream import ic


# Root dataset directory
dataset_path = os.path.expanduser('~/Desktop/UniToBrain')


device = torch.device('mps') or ('cuda' if torch.cuda.is_available() else 'cpu')

class PerfusionDataset(Dataset):
    def __init__(self, dataset_path, in_hu=True, window: tuple = None):
        self.dataset_path = dataset_path # UniToBrain folder containing dicom folders
        self.in_hu = in_hu # Whether the dicom images' pixel values are in Hounsfield Units
        self.window = window # (window_center, window_width) for windowing the images or None
        # NOTE: Temporary implementation, will be changed to filter out the irrelevant folders
        self.folder_paths = self.load_folder_paths()
        
    def __len__(self):
        return len(self.folders)
    
class NextFrameDataset(PerfusionDataset):
    """
    Provides data for next-frame prediction based only on the previous frame
    """
    def __getitem__(self, idx):
        volumes = DataTransformer.get_volume(self.folders[idx]) # (time, slices, height, width)
        x = volumes[:-1]
        y = volumes[1:]
        return x, y
    
class AttentionWindowDataset(PerfusionDataset):
    def __init__(self, root_dir, block_size: int, batch_size: int, in_hu=True, window: tuple = None):
        super().__init__(root_dir, in_hu, window)
        self.block_size = block_size
        self.batch_size = batch_size

    def __getitem__(self, idx):

        data = DataTransformer.get_volume(self.folders[idx])
        ix = np.random.sample
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y, = x.to(device), y.to(device)
        return x, y
        