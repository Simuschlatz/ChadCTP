from preprocessing import get_volume, load_folder_paths, get_2d_mask, get_largest_connected_component
from visualization import interactive_plot, multi_vol_seq_iplot,scroll_through_all_slices, \
    interactive_plot_with_threshold, interactive_plot_with_mask, interactive_plot_with_binary_mask, \
    interactive_plot_with_3d_mask
import numpy as np
import os
if __name__ == "__main__":
    dataset_path = os.path.expanduser('~/Desktop/UniToBrain')
    # folder_path = 'MOL-099_Registered_Filtered_3mm_20HU'
    # folder_path = os.path.join(dataset_path, folder_path)

    folder_paths = load_folder_paths()
    # print(len(folder_paths))
    # folder_path = folder_paths[10]

    # v_raw = DataTransformer.get_volume(folder_path, spatial_downsampling_factor=1)
    # filtered_slice = bilateral_filter(v_raw[0, 5], 5, 10)
    # v_raw = DataTransformer.bilateral_filter(v_raw[len(v_raw) // 2:len(v_raw) // 2 + 1, :2], 10, 1)
    # interactive_plot(v_raw, title="Raw")
    import matplotlib.pyplot as plt
    for folder_path in folder_paths[13:]: # [os.path.join(dataset_path, 'MOL-120')]:

        volume_seq = get_volume(folder_path, 
                                extract_brain=True,
                                windowing=True,
                                correct_motion=True,
                                spatial_downsampling_factor=2, 
                                temporal_downsampling_factor=7)
        # volume_seq2 = get_volume(folder_path + '_Registered_Filtered_3mm_20HU', 
        #                         extract_brain=False,
        #                         windowing=True,
        #                         correct_motion=False,
        #                         spatial_downsampling_factor=2, 
        #                         temporal_downsampling_factor=7)
        # multi_vol_seq_iplot([volume_seq, volume_seq2], ['Mine', 'UniToBrain Team'])

        volume_seq2 =  get_volume(folder_path, 
                                extract_brain=True,
                                windowing=True,
                                correct_motion=True,
                                spatial_downsampling_factor=1, 
                                temporal_downsampling_factor=7)
        multi_vol_seq_iplot([volume_seq, volume_seq2], ['Downsampled 2x', 'FullSize'])
        # multi_vol_seq_iplot([volume_seq, volume_seq2], ['Extracted Brain', 'All Volume'])
        # print(volume_seq.shape)
        # interactive_plot(volume_seq)
        # interactive_plot_with_threshold(volume_seq, title=folder_path.split('/')[-1])
        # interactive_plot_with_3d_mask(volume_seq, title=folder_path.split('/')[-1], apply_window=True, threshold_max=180)
    # v = DataTransformer.get_volume(folder_path, spatial_downsampling_factor=4)
    # scroll_through_all_slices(v_registered, title=folder_path.split('/')[-1])

    # from quality_metrics import snr
    # print(snr(v))
    # multiple_interactive_plots(v_registered_filtered_list, titles=[folder_path.split('/')[-1] for folder_path in folder_paths[3:5]], plotting_function=scroll_through_all_slices)
    # interactive_plot(v_raw, title="Raw")
    # multiple_interactive_plot([v_raw, v_registered_filtered])

    # interactive_plot(v_raw, title="FromDataset")
    # Select a single time point (e.g., the first one) for volume rendering
    # multi_vol_seq_iplot([v_raw[0, 5], filtered_slice])

    # Filtering the volume sequence
    # Copy the volume sequence to avoid modifying the original
    # filtered_volume_seq = v_raw.copy()
    # print(id)
    # filter_volume_seq(filtered_volume_seq, 10, 7)
    # multi_vol_seq_iplot([filtered_volume_seq, v_raw, v_registered_filtered], ['Filtered', 'Raw', 'Registered & Filtered'])
