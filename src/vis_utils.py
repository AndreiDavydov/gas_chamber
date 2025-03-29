import matplotlib.pyplot as plt
import numpy as np

def vis_grid(frames, r=5, c=5):
    take_every = len(frames) // (r*c)
    frames_vis = frames[::take_every]

    fig, ax = plt.subplots(r,c, figsize=(5*c,5*r))

    if r == 1:
        ax = ax[None,:]

    for i in range(r):
        for j in range(c):
            k = c * i + j
            ax[i,j].imshow(frames_vis[k])
            ax[i,j].set_axis_off()

    fig.tight_layout()
    plt.tight_layout(pad=0.00)
    return fig


def blend_w_future_masks(background_video, masks, num_future=5, alpha=0.4):
    """
    Blend the masks with the background image.

    Take only the first num_future frames from the masks. Blend them with the bakcground image using different colors.
    The blending is done by superimposing the masks on the background image, using the alpha parameter to control the blending.
    The masks colored in rainbow colors.

    background : np.array, rgb image : N x H x W x 3
    masks : np.array, masks of the video, stack of binary arrays, (N-1) x H x W

    For a new frame, colors must roll with the increment of 1, so that the masks on previous frames are colored in the same color.
    """
    # Create a color map
    num_colors = 30
    cmap = plt.get_cmap('hsv', num_colors)

    blended_video = np.zeros_like(background_video)

    # Iterate over the masks and blend them with the background
    for idx in range(len(background_video)):
        frame = background_video[idx].copy()
        masks_idx = masks[idx:]
        
        start_color = idx % num_colors
        colors = (np.arange(start_color, start_color + num_colors) % num_colors)[:num_future]
        for mask_i, mask in enumerate(masks_idx):
            color = cmap(colors[mask_i])[:3]
            color = np.array(color).reshape(1, 1, 3)
            frame[mask > 0] = frame[mask > 0] * (1 - alpha) + color * alpha
            
            if mask_i == num_future - 1: break
        
        blended_video[idx] = frame
    return blended_video