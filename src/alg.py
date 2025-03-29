import cv2
import numpy as np


def denoise_and_fill(x, open_kernel_size=(3,3), open_iterations=1, close_kernel_size=(5,5), close_iterations=3):
    """
    Denoises the image and fills small holes in the image
    Denoising includes the following morpholgy operations:
    1. Removing noise via opening with small kernel #(3,3)
    2. Filling small holes via closing with large kernel #(5,5)
    3. Remove larger noise via opening with large kernel #(5,5)
    """

    x = x.copy()
    # denoise: removes noise via opening
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, open_kernel_size)
    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel_open, iterations=open_iterations)
    
    # fill small holes via closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, close_kernel_size)
    x = cv2.morphologyEx(x, cv2.MORPH_CLOSE, kernel_close, iterations=close_iterations)

    x = cv2.morphologyEx(x, cv2.MORPH_OPEN, kernel_close, iterations=open_iterations)

    return x


def update_state(state, diff_raw, denoise_out=True, **denoise_kwargs):
    """
    Update the state with the new difference frame

    The state consists of clusters of non-zero pixels. Diff is the difference between the current frame and the previous frame, which also represents the clusters of change.

    For each cluster (component) of the diff, there are 4 situations: 
    1. component is fully inside x - do nothing
    2. component is fully outside x - add it to x
    3. component is mostly inside x - shrink x
    4. component is mostly outside x - expand x
    """

    x = denoise_and_fill(state, **denoise_kwargs)
    y = denoise_and_fill(diff_raw, **denoise_kwargs)
    
    # all intersection checks are done in binary space
    x_bin = (x > 0).astype(np.uint8)
    y_bin = (y > 0).astype(np.uint8)

    # split y in connected components
    num_labels, labels = cv2.connectedComponents(y_bin)
    comps = [(labels == i).astype(np.uint8) for i in range(1, num_labels)]
        
    for comp in comps:
        intersection = np.logical_and(x_bin, comp)
        union = np.logical_or(x_bin, comp)

        if intersection.sum() == comp.sum():
            # case 1: do nothing
            pass
        elif intersection.sum() == 0:
            # case 2: add it to x
            x = x + y * comp
        else:
            # case 3 or 4: shrink or expand x
            if intersection.sum() / union.sum() > 0.5:
                # case 3: shrink x
                comp = 1 - comp
                x = np.logical_and(x, comp)
            else:
                # case 4: expand x
                x = x + y * comp

    if denoise_out:
        x = denoise_and_fill(x, **denoise_kwargs)

    return x