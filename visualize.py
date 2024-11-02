"""
3D Slicer, FAST
"""
import scipy
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio
from skimage import measure
from icecream import ic
import numpy as np

def render_volume_slices(volume, cmap='magma', figsize=(12, 12)):
    """
    Renders all slices of a 3D volume with no gaps between images.
    
    Parameters:
    - volume: 3D numpy array representing the volume
    - cmap: colormap for the images (default: 'magma')
    - figsize: size of the figure (default: (12, 12))
    
    Returns:
    - fig: matplotlib figure object
    - axes: 2D array of matplotlib axes objects
    """
    depth = len(volume)
    rows = int(np.sqrt(depth))
    cols = int(np.ceil(depth / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    fig.subplots_adjust(wspace=0, hspace=0)
    
    for i, ax in enumerate(axes.flat):
        if i < depth:
            ax.imshow(volume[i], cmap=cmap, aspect='equal')
        ax.axis('off')
    
    plt.tight_layout()
    fig.patch.set_facecolor('black')
    plt.show(block=True)

def scroll_through_all_slices(volume_seq: np.ndarray, title="", show=True):
    t, y, z, x = volume_seq.shape
    # Reshape the 4D volume sequence into a 3D array
    volume = volume_seq.reshape((t * y, z, x))
    
    # Create the figure and axis
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    # Set the title
    plt.title(title)
    
    # Display the initial slice
    img = ax.imshow(volume[0], cmap='magma')
    plt.colorbar(img)
    
    # Create the slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, volume.shape[0] - 1, valinit=0, valstep=1)
    
    # Update function for the slider
    def update(val):
        slice_index = int(slider.val)
        img.set_data(volume[slice_index])
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    
    if show:
        plt.show()
    
    return fig, ax

def interactive_plot(volume_seq, title="", show=True):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    image = ax.imshow(volume_seq[0, 0], cmap='magma')
    plt.title(title)

    ax_slice_slider = plt.axes([0.25, 0.15, 0.65, 0.03])
    ax_time_slider = plt.axes([0.25, 0.1, 0.65, 0.03])

    time_slider = Slider(ax_time_slider, 'Time', 0, volume_seq.shape[0] - 1, valinit=0, valstep=1)
    slice_slider = Slider(ax_slice_slider, 'Slice', 0, volume_seq.shape[1] - 1, valinit=0, valstep=1)
    scrolling_slider = [time_slider]
    plt.colorbar(image, ax=ax)

    def update(val, scrolling_slider, slider):
        image.set_data(volume_seq[int(time_slider.val), int(slice_slider.val)])
        fig.canvas.draw_idle()
        scrolling_slider[0] = slider

    time_slider.on_changed(lambda val: update(val, scrolling_slider, time_slider))
    slice_slider.on_changed(lambda val: update(val, scrolling_slider, slice_slider))

    def on_scroll(event):
        if event.button == 'up':
            scrolling_slider[0].set_val(min(scrolling_slider[0].val + scrolling_slider[0].valstep, scrolling_slider[0].valmax))
        elif event.button == 'down':
            scrolling_slider[0].set_val(max(scrolling_slider[0].val - scrolling_slider[0].valstep, scrolling_slider[0].valmin))
        fig.canvas.draw_idle()

    def on_motion(event):
        if event.inaxes == ax:
            plt.gcf().canvas.set_cursor(1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    if show:
        plt.show(block=True)
    return fig, ax

def multiple_interactive_plots(volume_seqs, titles=None, plotting_function=interactive_plot):
    if titles is None:
        titles = [f"Volume {i+1}" for i in range(len(volume_seqs))]
    
    figures = []
    for volume_seq, title in zip(volume_seqs, titles):
        fig, ax = plotting_function(volume_seq, title, show=False)
        figures.append(fig)
    
    plt.show(block=False)
    
    input("Press Enter to close all windows...")
    for fig in figures:
        plt.close(fig)
    plt.ioff()  # Turn off interactive mode

def multi_vol_seq_iplot(volume_seqs, titles=None, nrows=None):
    """
    Renders multiple interactive plots in one scene with two unified sliders
    controlling all plots
    volume_seqs: iterable of 4D volume sequences.
    """
    plt.ion()  # Turn on interactive mode
    num_volumes = len(volume_seqs)

    if nrows is None:
        nrows = int(num_volumes ** .5)
    ncols = (num_volumes + nrows - 1) // nrows

    fig, axes = plt.subplots(nrows, 
                             ncols, 
                             figsize=(5*ncols, 5*nrows),
                             squeeze=True,
                             sharex='all',
                             sharey='all'
                             )
    plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

    ic(axes)
    if titles is None:
        titles = [f"Volume {i+1}" for i in range(num_volumes)]

    images = []
    for i, (volume_seq, title) in enumerate(zip(volume_seqs, titles)):
        # row, col = divmod(i, ncols)
        ax = axes[i] #if nrows == 1 else axes[row, col]
        image = ax.imshow(volume_seq[0, 0], cmap='magma')
        ax.set_title(title)
        plt.colorbar(image, ax=ax)
        images.append(image)

    ax_slice_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
    ax_time_slider = plt.axes([0.25, 0.05, 0.5, 0.03])

    max_time = max(vol.shape[0] for vol in volume_seqs) - 1
    max_slice = max(vol.shape[1] for vol in volume_seqs) - 1

    time_slider = Slider(ax_time_slider, 'Time', 0, max_time, valinit=0, valstep=1)
    slice_slider = Slider(ax_slice_slider, 'Slice', 0, max_slice, valinit=0, valstep=1)

    def update(val):
        for i, volume_seq in enumerate(volume_seqs):
            t = min(int(time_slider.val), volume_seq.shape[0] - 1)
            s = min(int(slice_slider.val), volume_seq.shape[1] - 1)
            images[i].set_data(volume_seq[t, s])
        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    slice_slider.on_changed(update)

    def on_scroll(event):
        if event.inaxes == ax_time_slider:
            time_slider.set_val(min(max(time_slider.val + event.step, time_slider.valmin), time_slider.valmax))
        elif event.inaxes == ax_slice_slider:
            slice_slider.set_val(min(max(slice_slider.val + event.step, slice_slider.valmin), slice_slider.valmax))
        else:
            for ax in axes.flat:
                if event.inaxes == ax:
                    slice_slider.set_val(min(max(slice_slider.val + event.step, slice_slider.valmin), slice_slider.valmax))
                    break
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    plt.show(block=True)

from preprocessing import get_mask

def interactive_plot_with_threshold(volume_seq, title="", show=True, min_thresh=-40, max_thresh=120, apply_window=True):

    """Like interactive_plot but with additional threshold sliders that show a binary mask"""
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9, wspace=0.3)
    
    # Original image on left
    if apply_window:
        image = ax1.imshow(np.clip(volume_seq[0, 0], a_min=-40, a_max=120), cmap='magma')
    else:
        image = ax1.imshow(volume_seq[0, 0], cmap='magma')
    ax1.set_title(f"{title} - Original")
    # plt.colorbar(image, ax=ax1)
    
    # Thresholded mask on right
    mask = ax2.imshow((volume_seq[0, 0] >= min_thresh) & (volume_seq[0, 0] <= max_thresh), cmap='binary')
    ax2.set_title(f"{title} - Threshold Mask")
    # Add legend (black = 0, white = 1)
    # ax2.legend(['0', '1'], loc='upper right')

    # Create sliders
    ax_slice = plt.axes([0.25, 0.25, 0.5, 0.03])
    ax_time = plt.axes([0.25, 0.2, 0.5, 0.03])
    ax_min_thresh = plt.axes([0.25, 0.15, 0.5, 0.03])
    ax_max_thresh = plt.axes([0.25, 0.1, 0.5, 0.03])

    data_min, data_max = np.min(volume_seq), np.max(volume_seq)
    
    time_slider = Slider(ax_time, 'Time', 0, volume_seq.shape[0]-1, valinit=0, valstep=1)
    slice_slider = Slider(ax_slice, 'Slice', 0, volume_seq.shape[1]-1, valinit=0, valstep=1)
    min_thresh_slider = Slider(ax_min_thresh, 'Min Threshold', data_min, data_max, valinit=min_thresh)
    max_thresh_slider = Slider(ax_max_thresh, 'Max Threshold', data_min, data_max, valinit=max_thresh)
    
    scrolling_slider = [time_slider]

    def update(val, scrolling_slider=None, slider=None):
        current_slice = volume_seq[int(time_slider.val), int(slice_slider.val)]
        if apply_window:
            image.set_data(np.clip(current_slice, a_min=-40, a_max=120))
        else:
            image.set_data(current_slice)
        # Update threshold mask
        thresh_mask = (current_slice >= min_thresh_slider.val) & (current_slice <= max_thresh_slider.val)
        mask.set_data(thresh_mask)
        
        
        fig.canvas.draw_idle()
        if scrolling_slider is not None and slider is not None:
            scrolling_slider[0] = slider

    # Connect update function to all sliders
    time_slider.on_changed(lambda val: update(val, scrolling_slider, time_slider))
    slice_slider.on_changed(lambda val: update(val, scrolling_slider, slice_slider))
    min_thresh_slider.on_changed(update)
    max_thresh_slider.on_changed(update)

    def on_scroll(event):
        if event.button == 'up':
            scrolling_slider[0].set_val(min(scrolling_slider[0].val + scrolling_slider[0].valstep, 
                                          scrolling_slider[0].valmax))
        elif event.button == 'down':
            scrolling_slider[0].set_val(max(scrolling_slider[0].val - scrolling_slider[0].valstep, 
                                          scrolling_slider[0].valmin))
        
        fig.canvas.draw_idle()

    def on_motion(event):
        if event.inaxes in [ax1, ax2]:
            plt.gcf().canvas.set_cursor(1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    if show:
        plt.show(block=True)
    return fig, (ax1, ax2)

def interactive_plot_with_mask(volume_seq, title="", show=True, initial_min_objects=750, threshold_min=-25, threshold_max=140, apply_window=True):
    """Like interactive_plot but shows the mask from get_mask with adjustable min_objects parameter"""
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
    plt.subplots_adjust(left=0.1, bottom=0.35, right=0.9, top=0.9, wspace=0.3)
    
    # Original image on left
    if apply_window:
        image = ax1.imshow(np.clip(volume_seq[0, 0], a_min=-40, a_max=120), cmap='magma')
    else:
        image = ax1.imshow(volume_seq[0, 0], cmap='magma')
    ax1.set_title(f"{title} - Original")
    
    # Mask in middle
    mask_img = get_mask(volume_seq[0, 0], 
                        threshold_min=threshold_min, 
                        threshold_max=threshold_max, 
                        remove_small_objects_size=initial_min_objects
                        )
    mask_plot = ax2.imshow(mask_img, cmap='binary')
    ax2.set_title(f"{title} - Brain Mask")

    # Masked image on right
    masked_img = volume_seq[0, 0] * mask_img
    if apply_window:
        masked_image = ax3.imshow(np.clip(masked_img, a_min=-40, a_max=120), cmap='magma')
    else:
        masked_image = ax3.imshow(masked_img, cmap='magma')
    ax3.set_title(f"{title} - Masked Brain")

    # Create sliders
    ax_slice = plt.axes([0.25, 0.2, 0.5, 0.03])
    ax_time = plt.axes([0.25, 0.17, 0.5, 0.03]) 
    ax_min_objects = plt.axes([0.25, 0.14, 0.5, 0.03])
    ax_thresh_min = plt.axes([0.25, 0.11, 0.5, 0.03])
    ax_thresh_max = plt.axes([0.25, 0.08, 0.5, 0.03])
    
    time_slider = Slider(ax_time, 'Time', 0, volume_seq.shape[0]-1, valinit=0, valstep=1)
    slice_slider = Slider(ax_slice, 'Slice', 0, volume_seq.shape[1]-1, valinit=0, valstep=1)
    min_objects_slider = Slider(ax_min_objects, 'Min Objects Size', 0, 2000, valinit=initial_min_objects)
    thresh_min_slider = Slider(ax_thresh_min, 'Threshold Min', -100, 200, valinit=threshold_min, valstep=1)
    thresh_max_slider = Slider(ax_thresh_max, 'Threshold Max', -100, 400, valinit=threshold_max, valstep=1)

    def update(val):
        current_slice = volume_seq[int(time_slider.val), int(slice_slider.val)]
        if apply_window:
            image.set_data(np.clip(current_slice, a_min=-40, a_max=120))
        else:
            image.set_data(current_slice)
            
        # Update mask with threshold parameters
        mask_img = get_mask(current_slice, 
                          threshold_min=thresh_min_slider.val,
                          threshold_max=thresh_max_slider.val,
                          remove_small_objects_size=int(min_objects_slider.val))
        mask_plot.set_data(mask_img)
        
        # Update masked image
        masked_img = current_slice * mask_img
        if apply_window:
            masked_image.set_data(np.clip(masked_img, a_min=-40, a_max=120))
        else:
            masked_image.set_data(masked_img)
        
        fig.canvas.draw_idle()

    # Connect update function to all sliders
    time_slider.on_changed(update)
    slice_slider.on_changed(update)
    min_objects_slider.on_changed(update)
    thresh_min_slider.on_changed(update)
    thresh_max_slider.on_changed(update)

    def on_scroll(event):
        if event.inaxes == ax_time:
            slider = time_slider
        elif event.inaxes == ax_min_objects:
            slider = min_objects_slider
        elif event.inaxes == ax_thresh_min:
            slider = thresh_min_slider
        elif event.inaxes == ax_thresh_max:
            slider = thresh_max_slider
        else:
            slider = slice_slider
            
        if event.button == 'up':
            slider.set_val(min(slider.val + slider.valstep, slider.valmax))
        elif event.button == 'down':
            slider.set_val(max(slider.val - slider.valstep, slider.valmin))

    def on_motion(event):
        if event.inaxes in [ax1, ax2, ax3]:
            plt.gcf().canvas.set_cursor(1)

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    if show:
        plt.show(block=True)
    return fig, (ax1, ax2, ax3)

from matplotlib import animation, rc

def create_animation(ims):
    rc('animation', html='jshtml')
    fig = plt.figure(figsize=(6, 6))
    plt.axis('off')
    im = plt.imshow(ims[0], cmap="gray")

    def animate_func(i):
        im.set_array(ims[i])
        return [im]

    return animation.FuncAnimation(fig, animate_func, frames = len(ims), interval = 1000//24)

def downsample_volume(volume, factor):
    """Downsamples the volume by the given factor."""
    return scipy.ndimage.zoom(volume, (1/factor, 1/factor, 1/factor), order=1)


def plot_3d_skeleton(ctvol, outprefix='./'):
    """
    Make a 3D plot of the skeleton.
    <units> either 'HU' or 'normalized' determines the thresholds
    """
    #from https://www.kaggle.com/gzuidhof/full-preprocessing-tutorial
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = ctvol.transpose(2,1,0)
    p = np.flip(p, axis = 0) #need this line or else the patient is upside-down
    
    #https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.marching_cubes_lewiner
    verts, faces, _ignore1, _ignore2 = measure.marching_cubes(p)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    plt.savefig(outprefix+'_3D_Bones.png')
    plt.close()    

def make_gifs(ctvol, outprefix, chosen_views):
    """Save GIFs of the <ctvol> in the axial, sagittal, and coronal planes.
    This assumes the final orientation produced by the preprocess_volumes.py
    script: [slices, square, square].
    
    <chosen_views> is a list of strings that can contain any or all of
        ['axial','coronal','sagittal']. It specifies which view(s) will be
        made into gifs."""
    #https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in-python
    
    #First fix the grayscale colors.
    #imageio assumes only 256 colors (uint8): https://stackoverflow.com/questions/41084883/imageio-how-to-increase-quality-of-output-gifs
    #If you do not truncate to a 256 range, imageio will do so on a per-slice
    #basis, which creates weird brightening and darkening artefacts in the gif.
    #Thus, before making the gif, you should truncate to the 0-256 range
    #and cast to a uint8 (the dtype imageio requires):
    #how to truncate to 0-256 range: https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
    ctvol = np.clip(ctvol, a_min=-800, a_max=400)
    ctvol = (  ((ctvol+800)*(255))/(400+800)  ).astype('uint8')
    
    #Now create the gifs in each plane
    if 'axial' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[0]):
            images.append(ctvol[slicenum,:,:])
        imageio.mimsave(outprefix+'_axial.gif',images)
        print('\t\tdone with axial gif')
    
    if 'coronal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[1]):
            images.append(ctvol[:,slicenum,:])
        imageio.mimsave(outprefix+'_coronal.gif',images)
        print('\t\tdone with coronal gif')
    
    if 'sagittal' in chosen_views:
        images = []
        for slicenum in range(ctvol.shape[2]):
            images.append(ctvol[:,:,slicenum])
        imageio.mimsave(outprefix+'_sagittal.gif',images)
        print('\t\tdone with sagittal gif')


# 0 -> m -
