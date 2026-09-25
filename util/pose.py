"""
Video handling + sequence cleaning helper - MediaPipe backend-er jonno.
Pose/hand detection nijei (util/mp_pose.py) kore, ei file-ta shudhu generic
kaj-gulo kore, jegulo joint-sonkha (75, ba onno kono) niye matha ghamay na:

    video --(iter_sampled_frames)--> frames
          --(util/mp_pose.py: extract_sequence_mp)--> raw keypoints (T, 75, 3)
          --(clean_sequence)--> interpolated, NaN-free sequence
          --(sliding_windows)--> (N, num_frame, 75, 3)  for ST-GCN

x, y are stored normalised (0..1) - MediaPipe nijei ei format-e dey.
"""
import cv2
import numpy as np


# --------------------------------------------------------------------------- #
#  Video reading
# --------------------------------------------------------------------------- #
def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not fps or fps != fps or fps < 1:  # 0 / NaN
        fps = 30.0
    return fps, w, h


def iter_sampled_frames(video_path, target_fps=30):
    """Yield (frame_index, BGR frame). If the video fps is higher than
    target_fps, frames are dropped so the output is ~target_fps."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f'Cannot open video: {video_path}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0
    interval = max(fps / target_fps, 1.0) if target_fps else 1.0
    idx, next_pick = 0, 0.0
    try:
        while True:
            if not cap.grab():
                break
            if idx >= next_pick - 1e-6:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                yield idx, frame
                next_pick += interval
            idx += 1
    finally:
        cap.release()


# --------------------------------------------------------------------------- #
#  Cleaning
# --------------------------------------------------------------------------- #
def clean_sequence(seq, kpt_thr=0.2, min_valid_ratio=0.3):
    """Fill missing frames / low-confidence joints by temporal interpolation.
    seq: (T, V, 3) - V jekono joint-sonkha hote pare (75 for body_hand).
    Returns float32 (T,V,3) without NaN, or None if the video is unusable."""
    seq = seq.astype(np.float32).copy()
    T = len(seq)
    if T == 0:
        return None
    detected = ~np.isnan(seq[:, :, 0]).all(axis=1)
    if detected.mean() < min_valid_ratio:
        return None

    conf = seq[:, :, 2]
    low = np.nan_to_num(conf, nan=0.0) < kpt_thr
    xy = seq[:, :, :2]
    xy[low] = np.nan                               # in-place on view
    seq[:, :, 2] = np.nan_to_num(conf, nan=0.0)    # missing frame -> conf 0

    t = np.arange(T)
    for j in range(seq.shape[1]):
        for c in range(2):
            v = seq[:, j, c]
            ok = ~np.isnan(v)
            if ok.any():
                seq[:, j, c] = np.interp(t, t[ok], v[ok])   # edges = nearest value

    # joint that was never visible in the whole video -> mean of visible joints
    nan_mask = np.isnan(seq[:, :, :2])
    if nan_mask.any():
        with np.errstate(all='ignore'):
            mean_xy = np.nanmean(seq[:, :, :2], axis=1, keepdims=True)
        seq[:, :, :2] = np.where(nan_mask, mean_xy, seq[:, :, :2])
    if np.isnan(seq).any():
        return None
    return seq


# --------------------------------------------------------------------------- #
#  Windows
# --------------------------------------------------------------------------- #
def sliding_windows(seq, n_frames=30, stride=10):
    """(T,V,3) -> (N, n_frames, V, 3) and list of start indices.
    Short videos are padded by repeating the last frame."""
    T = len(seq)
    if T < n_frames:
        pad = np.repeat(seq[-1:], n_frames - T, axis=0)
        seq = np.concatenate([seq, pad], axis=0)
        T = n_frames
    starts = list(range(0, T - n_frames + 1, stride))
    if starts[-1] != T - n_frames:
        starts.append(T - n_frames)                # cover the tail
    wins = np.stack([seq[s:s + n_frames] for s in starts], axis=0)
    return wins.astype(np.float32), starts
