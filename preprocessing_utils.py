import os
import pickle
from icecream import ic

dataset_path = os.path.expanduser('~/Desktop/UniToBrain')

def save_folder_paths(output_file: str='folder_paths.pkl'):
    folder_paths = []
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if len(folder) == 7 and len(os.listdir(folder_path)) != 288: # MOL-XYZ
            folder_paths.append(folder_path)
    ic(len(folder_paths), folder_paths[len(folder_paths)-10:])

    output_file = os.path.join(dataset_path, output_file)
    with open(output_file, 'wb') as f:
        pickle.dump(folder_paths, f)

    ic(f"Folder paths saved to {output_file}")
    return folder_paths

def load_folder_paths(output_file: str='folder_paths_288.pkl'):
    file_path = os.path.join(dataset_path, output_file)
    if not os.path.exists(file_path):
        print(f"File {output_file} does not exist, running save_folder_paths() instead...")
        return save_folder_paths(output_file=output_file)
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)
