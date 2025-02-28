import os
import numpy as np

import src
import src.preproc_utils as prep
import example as ex

CUT = False
ALIGN = False
GRAY = False
MASK_PREP = True
MASK_FRAMES = True

def _cut():
    video_path = os.path.join(src.DATAPATH, "raw/MVI_0022.MP4")
    raw_frames = src.video_utils.get_frames(video_path)

    print("Preproc Step 1: cut the video to the relevant part")
    cut_frames = prep.cut_video_to_bbox(raw_frames, ex.BBOX_TO_CUT, ex.NUM_FRAMES_TO_SKIP)
    # save video
    cut_path = os.path.join(src.DATAPATH, "processed/cut.MP4")
    src.utils.write_frames(cut_frames, cut_path)
    return cut_frames

def _align(cut_frames=None):
    if not CUT:
        cut_frames = src.video_utils.get_frames(os.path.join(src.DATAPATH, "processed/cut.MP4"))
    print("Preproc Step 2: align the frames")
    aligned_frames = prep.align_frames(cut_frames, bbox_to_align=ex.BBOX_TO_ALIGN)
    # save video
    aligned_path = os.path.join(src.DATAPATH, "processed/cut_aligned.MP4")
    src.utils.write_frames(aligned_frames, aligned_path)
    return aligned_frames

def _gray(aligned_frames=None):
    if not ALIGN:
        aligned_frames = src.video_utils.get_frames(os.path.join(src.DATAPATH, "processed/cut_aligned.MP4"))
    print("Preproc Step 3: convert to grayscale")
    frames_gray = prep.rgb2gray(aligned_frames)
    # save video
    aligned_gray_path = os.path.join(src.DATAPATH, "processed/cut_aligned_gray.MP4")
    src.utils.write_frames(frames_gray, aligned_gray_path)

    return frames_gray

def _mask(frames_gray=None, vis=True):
    if not GRAY:
        frames_gray = src.video_utils.get_frames(os.path.join(src.DATAPATH, "processed/cut_aligned_gray.MP4"))
        frames_gray = prep.rgb2gray(frames_gray)
    print("Preproc Step 4: Prepare the binary mask of the snake tube")

    mask = prep.get_mask_from_points(frames_gray[0].shape, ex.SNAKE_ANCHOR_POINTS)
    # save mask
    mask_path = os.path.join(src.DATAPATH, "processed/mask.npy")
    np.save(mask_path, mask)

    # vis mask
    if vis:
        mask_vis_path = os.path.join(src.DATAPATH, "processed/mask_vis.png")
        src.utils.vis_mask(frames_gray[0], mask, ex.SNAKE_ANCHOR_POINTS, mask_vis_path)
    return mask, frames_gray


def _mask_frames(mask=None, frames_gray=None):
    if not MASK_PREP:
        mask = np.load(os.path.join(src.DATAPATH, "processed/mask.npy"))
        frames_gray = src.video_utils.get_frames(os.path.join(src.DATAPATH, "processed/cut_aligned_gray.MP4"))
        frames_gray = prep.rgb2gray(frames_gray)

    print("Preproc Step 5: Apply mask to all frames")
    frames_masked = prep.mask_frames(mask, frames_gray)
    
    # save video
    mask_path = os.path.join(src.DATAPATH, "processed/cut_aligned_gray_masked.MP4")
    src.utils.write_frames(frames_masked, mask_path)
    return frames_masked


if __name__ == "__main__":
    cut_frames = _cut() if CUT else None
        
    aligned_frames = _align(cut_frames) if ALIGN else None

    frames_gray = _gray(aligned_frames) if GRAY else None

    mask, frames_gray = _mask(frames_gray) if MASK_PREP else (None, None)

    frames_masked = _mask_frames(mask, frames_gray) if MASK_FRAMES else None

    