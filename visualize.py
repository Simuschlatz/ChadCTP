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

def multiple_interactive_plots(volume_seqs, titles=None):
    if titles is None:
        titles = [f"Volume {i+1}" for i in range(len(volume_seqs))]
    
    figures = []
    for volume_seq, title in zip(volume_seqs, titles):
        fig, ax = interactive_plot(volume_seq, title, show=False)
        figures.append(fig)
    
    plt.show(block=False)
    
    input("Press Enter to close all windows...")
    for fig in figures:
        plt.close(fig)
    plt.ioff()  # Turn off interactive mode


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
