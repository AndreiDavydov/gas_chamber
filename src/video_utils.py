import os
import random

import subprocess
import tempfile

from concurrent.futures import ThreadPoolExecutor

import cv2
import ffmpeg
import imageio
import numpy as np


def get_frames_cv2(vid_file, scale=1):
    """
    Read video frames from a file
    """

    video = cv2.VideoCapture(vid_file)
    if video.isOpened() is False:
        print("Error opening video file.")
        exit()
    fps = video.get(cv2.CAP_PROP_FPS)

    frame_list = []
    while video.isOpened():
        ret, frame = video.read()
        if ret is False:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if scale != 1:
            img = cv2.resize(
                img, (int(img.shape[1] * scale), int(img.shape[0] * scale))
            )
        frame_list.append(img)
    video.release()

    return frame_list, fps


def write_video_cv2(res_frames, out_path, fps=30):
    """
    Write frames to a video
    """
    width = res_frames[0].shape[1]
    height = res_frames[0].shape[0]

    tmp_file = out_path
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_file, fourcc, fps, (width, height))
    for frame in res_frames:
        out.write(frame[:, :, ::-1])
    out.release()

    print(f"Video is saved to '{tmp_file}'")


def concat_videos_cv2(video_paths, out_path="/tmp/concat_vid.mp4", fps=20, scale=1):
    """
    Stacks all videos frame-by-frame horizontally
    NOTE: All video frames must have the same shapes and #frames!
    """

    vid_frames = []
    for vid_path in video_paths:
        frame_list, _ = get_frames_cv2(vid_path, scale=scale)
        vid_frames.append(frame_list)

    num_frames = len(frame_list)

    out_frames = []
    for frame_idx in range(num_frames):
        out_frame = [vid[frame_idx] for vid in vid_frames]
        out_frame = np.concatenate(out_frame, axis=1)
        out_frames.append(out_frame)

    write_video_cv2(out_frames, out_path, fps=fps)


def write_video_gif(res_frames, fps=30):
    """
    Write frames to a video in a gif format
    """
    tmp_file = tempfile.mkdtemp() + ".gif"
    imageio.mimsave(tmp_file, res_frames, format="GIF", fps=fps)
    print(f"Video is saved to '{tmp_file}'")


def write_video_slow(res_frames, out_path, fps=30, crf=20):
    folder = tempfile.mkdtemp()
    for i, frame in enumerate(res_frames):
        f = os.path.join(folder, f"{i:06d}.png")
        imageio.imsave(f, frame)

    glob_path = f"{folder}/*.png"
    out = ffmpeg.input(glob_path, pattern_type="glob", r=fps)
    video = (
        out.output(out_path, pix_fmt="yuv420p", crf=crf, loglevel="quiet")
        .overwrite_output()
        .run_async()
    )
    video.wait()


def write_video_from_list(list_of_frames, out_path, fps=30, crf=20):
    _check_dir_exists(out_path)

    ### save list to the txt file
    tmp_dir = _get_tmp_dir()
    tmp_txt = os.path.join(tmp_dir, "list.txt")
    with open(tmp_txt, "w") as f:
        for frame in list_of_frames:
            f.write(f"file {frame}\n")

    out = ffmpeg.input(tmp_txt, r=fps, f="concat", safe="0")
    video = (
        out.output(out_path, pix_fmt="yuv420p", crf=crf, loglevel="quiet")
        .overwrite_output()
        .run_async()
    )
    video.wait()
    _rm_dir(tmp_dir)


def write_video_from_glob(glob_mask, out_path, fps=30, crf=20):
    _check_dir_exists(out_path)

    out = ffmpeg.input(glob_mask, pattern_type="glob", r=fps)
    video = (
        out.output(out_path, pix_fmt="yuv420p", crf=crf, loglevel="quiet")
        .overwrite_output()
        .run_async()
    )
    video.wait()


def write_video_from_folder(folder, out_path, fps=30, crf=20, verbose=False):
    _check_dir_exists(out_path)

    if verbose:
        print(f"Frames are pre-saved to {folder}...")
    glob_path = os.path.join(folder, "*.png")
    write_video_from_glob(glob_path, out_path, fps=fps, crf=crf)
    return glob_path


def write_video(res_frames, out_path, fps=30, crf=20, verbose=False):
    _check_dir_exists(out_path)

    assert (
        res_frames[0].shape[0] % 2 == 0 and res_frames[0].shape[1] % 2 == 0
    ), "H and W must be even to write with ffmpeg! #TODO"

    if verbose:
        print(f"Write video to {out_path}...")
    # import care.data.io.typed as typed
    import matplotlib.pyplot as plt

    folder = tempfile.mkdtemp()
    with ThreadPoolExecutor(max_workers=32) as save_pool:
        for i, frame in enumerate(res_frames):
            f = os.path.join(folder, f"{i:06d}.png")
            # save_pool.submit(typed.save, f, frame)
            save_pool.submit(plt.imsave, f, frame)

    glob_path = write_video_from_folder(
        folder, out_path, fps=fps, crf=crf, verbose=verbose
    )

    ### folder cleanup
    subprocess.call(f"rm -r {folder}".split(" "))

    if verbose:
        print(f"Video is saved to {out_path}!")
    return glob_path, out_path


def get_frames(in_path):
    probe = ffmpeg.probe(in_path)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    width = int(video_info["width"])
    height = int(video_info["height"])
    num_frames = int(video_info["nb_frames"])

    out, err = (
        ffmpeg.input(in_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24", loglevel="quiet")
        .run(capture_stdout=True)
    )

    frames = np.frombuffer(out, np.uint8).reshape([-1, height, width, 3])
    return frames


def stack_videos(video_files, output_video_path, axis=1):
    """
    video_files : list of .mp4 files to stack.
    axis == 0 - stack vertically   (vstack in ffmpeg)
    axis == 1 - stack horizontally (hstack in ffmpeg)

    Inspired by https://stackoverflow.com/questions/11552565/vertically-or-horizontally-stack-mosaic-several-videos-using-ffmpeg
    """
    _check_dir_exists(output_video_path)

    for f in video_files:
        assert os.path.exists(f), f"video file {f} does not exist!"

    assert axis in [0, 1]

    out = ["ffmpeg"]
    for f in video_files:
        out.extend(f"-i {f}".split(" "))
    stack = "hstack" if axis == 1 else "vstack"
    out.append("-filter_complex")
    out.append(f"{stack}=inputs={len(video_files)}")
    out.append(output_video_path)
    out.extend("-loglevel quiet".split(" "))
    out.append("-y")  # overwrite - Yes
    subprocess.call(out)


def stack_videos_grid(video_files, output_video_path):
    """
    video_files : list of lists of .mp4 files to stack.
    """
    assert len(video_files) > 0
    for v_list in video_files:
        assert len(v_list) > 1, "For a line of videos, use stack_videos"

    lines = []
    tmp_dir = _get_tmp_dir()
    for i, v_list in enumerate(video_files):
        v = os.path.join(tmp_dir, f"{i:04d}.mp4")
        stack_videos(v_list, v, axis=1)
        lines.append(v)

    stack_videos(lines, output_video_path, axis=0)
    _rm_dir(tmp_dir)


def merge_videos(video_files, output_video_path):
    """
    Stack videos in temporal axis, one **after** another.
    """
    file_list = "_tmp_video_list.txt"
    with open(file_list, "w") as f:
        for video in video_files:
            print(f"file {video}", file=f)

    ffmpeg.input(file_list, format="concat", safe=0).output(
        output_video_path, c="copy", loglevel="quiet"
    ).overwrite_output().run()

    subprocess.call(["rm", f"{file_list}"])


def resize_video(video_file, video_file_new, h=None, w=None):
    if h is None and w is None:
        raise ValueError("One of new dimensions must be defined!")

    if h is not None and w is not None:
        raise NotImplementedError

    ### either h or w is known
    if h is None:
        h = -1

    if w is None:
        w = -1

    out = f"ffmpeg -y -loglevel quiet".split(" ")
    out.extend(f"-i {video_file} -vf scale={w}:{h} {video_file_new}".split(" "))

    subprocess.call(out)


def split_in_halves(video_file, out1, out2, axis):
    """
    splits in two halves, top/bottom (axis=0) or left/right (axis=1)

    Inspired by https://stackoverflow.com/questions/52582215/commands-to-cut-videos-in-half-horizontally-or-vertically-and-rejoin-them-later
    """
    top = "[0]crop=iw:ih/2:0:0[top]"
    bottom = "[0]crop=iw:ih/2:0:oh[bottom]"
    left = "[0]crop=iw/2:ih:0:0[left]"
    right = "[0]crop=iw/2:ih:ow:0[right]"

    out = ["ffmpeg"]
    out.extend(f"-i {video_file} -filter_complex".split(" "))
    if axis == 0:
        out.append(top + ";" + bottom)
        out.extend(f"-map [top] {out1}".split(" "))
        out.extend(f"-map [bottom] {out2}".split(" "))
    elif axis == 1:
        out.append(left + ";" + right)
        out.extend(f"-map [left] {out1}".split(" "))
        out.extend(f"-map [right] {out2}".split(" "))
    out.extend("-loglevel quiet".split(" "))
    out.append("-y")

    subprocess.call(out)


def _get_tmp_dir():
    n = f"__tmp_frames_{int(random.random() * int(1e9))}"
    tmp_dir = f"/tmp/{n}"
    os.makedirs(tmp_dir, exist_ok=True)
    return tmp_dir


def _rm_dir(folder):
    subprocess.call(f"rm -r {folder}".split(" "))


def _check_dir_exists(save_path):
    assert os.path.exists(os.path.dirname(save_path))


if __name__ == "__main__":
    x = np.zeros((100, 256, 256), dtype=np.uint8)
    out_path = "_test_.mp4"
    write_video(x, out_path, fps=30, verbose=True)