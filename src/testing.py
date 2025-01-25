from preprocessing import get_volume, load_folder_paths, get_2d_mask, get_largest_connected_component
from visualization import interactive_plot, multi_vol_seq_iplot,scroll_through_all_slices, \
    interactive_plot_with_threshold, interactive_plot_with_mask, interactive_plot_with_binary_mask, \
    interactive_plot_with_3d_mask, overlay_volumes, overlay_volume_sequence, interactive_plot_with_bilateral_filter
import numpy as np
import os
if __name__ == "__main__":
    dataset_path = os.path.expanduser('~/Desktop/UniToBrain')

    folder_paths = load_folder_paths('small')
    for folder_path in folder_paths: # [os.path.join(dataset_path, 'MOL-002')]: # folder_paths[13:]: 
        
        volume_seq = get_volume(folder_path, 
                                extract_brain=False,
                                windowing=False,
                                correct_motion=False,
                                filter=False,
                                standardize=False,
                                spatial_downsampling_factor=2, 
                                temporal_downsampling_factor=8)
                                
        interactive_plot_with_bilateral_filter(volume_seq, title=folder_path.split('/')[-1], windowing_params=(40, 80))
        # volume_seq = get_volume(folder_path, 
        #                         extract_brain=True,
        #                         windowing=True,
        #                         correct_motion=False,
        #                         filter=True,
        #                         spatial_downsampling_factor=1, 
        #                         temporal_downsampling_factor=80)
        # v2 = get_volume(folder_path + '_Registered_Filtered_3mm_20HU', 
        #                 extract_brain=False,
        #                 windowing=True,
        #                 correct_motion=False,
        #                 filter=False,
        #                 spatial_downsampling_factor=1, 
        #                 temporal_downsampling_factor=80)
        # interactive_plot(volume_seq)
        # multi_vol_seq_iplot([volume_seq, v2], ['Vorgestellt', 'UniToBrain Preprocessing'])
    # for folder_path in folder_paths[13:]: # [os.path.join(dataset_path, 'MOL-120')]:
        # overlay_volume_sequence(np.concatenate([volume_seq, v2[1:]], axis=0))

        # volume_seq = get_volume(folder_path, 
        #                         extract_brain=True,
        #                         windowing=True,
        #                         correct_motion=False,
        #                         filter=False,
        #                         spatial_downsampling_factor=2, 
        #                         temporal_downsampling_factor=1)
        # volume_seq2 = get_volume(folder_path, 
        #                         extract_brain=False,
        #                         windowing=True,
        #                         correct_motion=False,
        #                         filter=False,
        #                         spatial_downsampling_factor=2, 
        #                         temporal_downsampling_factor=7)
        # multi_vol_seq_iplot([volume_seq, volume_seq2], ['Preprocessed', 'Original'])

        # volume_seq2 =  get_volume(folder_path, 
        #                         extract_brain=True,
        #                         windowing=True,
        #                         correct_motion=True,
        #                         spatial_downsampling_factor=1, 
        #                         temporal_downsampling_factor=7)
        # multi_vol_seq_iplot([volume_seq, volume_seq2], ['Downsampled 2x', 'FullSize'])
        # multi_vol_seq_iplot([volume_seq, volume_seq2], ['Extracted Brain', 'All Volume'])
        # print(volume_seq.shape)
        # interactive_plot(volume_seq, title=folder_path.split('/')[-1])
        # interactive_plot_with_threshold(volume_seq, title=folder_path.split('/')[-1])
        # interactive_plot_with_3d_mask(volume_seq, title=folder_path.split('/')[-1], apply_window=True, threshold_max=180)
