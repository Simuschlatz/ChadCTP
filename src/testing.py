from preprocessing import get_volume, load_folder_paths, get_2d_mask, get_largest_connected_component
from visualization import interactive_plot, multi_vol_seq_iplot,scroll_through_all_slices, \
    interactive_plot_with_threshold, interactive_plot_with_mask, interactive_plot_with_binary_mask, \
    interactive_plot_with_3d_mask, overlay_volumes, overlay_volume_sequence, interactive_plot_with_bilateral_filter, \
    interactive_plot_cycle_from_folder, multi_folder_cycle_iplot_all
import numpy as np
import os
import pickle
        
if __name__ == "__main__":

    # multi_folder_cycle_iplot_all(["TestScans/" + sub for sub in os.listdir("TestScans")], windowing_params=(40, 80))
    multi_folder_cycle_iplot_all(['Experiments/Data', 'Experiments/Data2'], windowing_params=(40, 80))
    # v = np.load("TestScans/1/MOL-063.npy")
    # print(v.shape)
    # interactive_plot_cycle_from_folder(data_folder="Experiments/Data", json_path="Experiments/volumes.json")
    # dataset_path = os.path.expanduser('~/Desktop/UniToBrain')
    # scan_ids = ["/MOL-" + s for s in ['063', '092', '098', '104', '133']]
    # paths = [dataset_path + sid for sid in scan_ids]
    # with open("registration_scans.pkl", "rb") as f:
        # scans = pickle.load(f)[:len(scan_ids)]

    # folder_paths = load_folder_paths('small')
    # for folder_path in folder_paths: 
    
        # v1 = get_volume(folder_path, 
        #                 extract_brain=True,
        #                 correct_motion=True,
        #                 window_params=(200, 400),
        #                 filter=False,
        #                 standardize=True,
        #                 spatial_downsampling_factor=2, 
        #                 temporal_downsampling_factor=2)
        # interactive_plot(v1)

    # data_path = "Experiments/Data"
    # check = ['206', '251', '263', '100', '92', '106', '112', '099']
    # for scan_name in os.listdir(data_path):
    #     if scan_name[-7:-4] not in check: continue
    #     scan_path = os.path.join(data_path, scan_name)
    #     v = np.load(scan_path)
    #     interactive_plot(v, title=scan_name, windowing_params=(0, 4))
