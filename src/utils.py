import numpy as np
import matplotlib.pyplot as plt

import src


def convert_float_vals_to_images(frames):
    min_ = frames.min()
    max_ = frames.max()
    frames = (frames - min_) / (max_ - min_)
    frames = (frames * 255).astype(np.uint8)
    return frames

def write_frames(frames, out_path):
    """
    Same as src.video_utils.write_video but with normalization
    """
    v = convert_float_vals_to_images(frames)
    src.video_utils.write_video(v, out_path)
    return

def vis_mask(img, mask, points, out_path):
    """
    Visualize the mask on the frame
    """
    fig, ax = plt.subplots(1, 2, figsize=(14,7))
    ax[0].imshow(img)
    ax[0].fill(points[:,0], points[:,1], c="pink", alpha=0.6)
    ax[0].scatter(points[:,0], points[:,1], c=np.arange(len(points)), cmap="jet")
    ax[1].imshow(mask)
    fig.tight_layout()
    plt.subplots_adjust()
    plt.savefig(out_path)
    plt.close()
    return
