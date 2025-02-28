import src
import cv2
from tqdm import tqdm
import numpy as np
from skimage import draw


def cut_video_to_bbox(frames, bbox_to_cut, num_frames_to_skip=0):
    cut = frames[num_frames_to_skip:, bbox_to_cut[0]:bbox_to_cut[1], bbox_to_cut[2]:bbox_to_cut[3]]
    return cut

    
def align_frames(frames, bbox_to_align):
    '''
    Align frames using ECC algorithm by one little patch of the frame

    '''
    aligned_frames = []
    reference_frame = frames[0]
    reference_frame_cut = reference_frame[bbox_to_align[0]:bbox_to_align[1], bbox_to_align[2]:bbox_to_align[3]]
    reference_gray = cv2.cvtColor(reference_frame_cut, cv2.COLOR_BGR2GRAY)
    for frame in tqdm(frames):
        frame_cut = frame[bbox_to_align[0]:bbox_to_align[1], bbox_to_align[2]:bbox_to_align[3]]
        current_gray = cv2.cvtColor(frame_cut, cv2.COLOR_BGR2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-10)
        cc, warp_matrix = cv2.findTransformECC(reference_gray, current_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        aligned_frame = cv2.warpAffine(frame, warp_matrix, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        aligned_frames.append(aligned_frame)
    return np.array(aligned_frames)


def rgb2gray(rgb):
    rgb = np.dot(rgb[...,:3] / 255, [0.2989, 0.5870, 0.1140])
    return rgb

def get_mask_from_points(img_shape, points):
    '''
    points: np.array of shape (n_points, 2), ordered as (x, y)
    '''
    mask = draw.polygon2mask(img_shape, points[:, ::-1]) # y first
    mask = mask.astype(int)

    return mask

def mask_frames(mask, frames):
    frames_masked = frames.copy()
    for i in tqdm(range(len(frames_masked))):
        frames_masked[i][mask == 0] = 0.
    return frames_masked