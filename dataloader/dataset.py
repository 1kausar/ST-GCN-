import os
import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset

# body_hand (75 = 33 MediaPipe Pose + 21 left hand + 21 right hand): left <-> right
# joint swap (horizontal flip augmentation-er jonno).
# Pose part (0-32): protita L/R landmark pair swap hoy, node 0 (nose) jaiga-e thake.
# Hand part: purota left-hand block (33-53) purota right-hand block (54-74)-r sathe
# swap hoy - hand-er bhitor re-order lage na, karon x-coordinate mirror korle ekta
# left hand ekoi 21-ta landmark index-e right hand hoye jay.
FLIP_IDX = ([0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
            18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31]
           + list(range(54, 75)) + list(range(33, 54)))


# --------------------------------------------------------------------------- #
#  Normalisation
# --------------------------------------------------------------------------- #
def processing_data(features, mode='frame'):
    """Normalise xy to [-1, 1].

    features : (N, T, V, 2)  (or (T, V, 2) / (V, 2))

    mode='frame'  : original repo behaviour - every frame is min-max scaled by
                    its own joints (x and y independently). Body-er position
                    ar size frame-by-frame muche jay.
    mode='window' : whole 30-frame window-er ekta common bounding box diye
                    scale kore, aspect ratio thik rekhe. Body-er absolute
                    movement (jump-e upore-nichey jaoa, running-e agiye jaoa)
                    motion stream-e theke jay.
    """
    xy = np.asarray(features, dtype=np.float32)
    if xy.ndim == 2:
        xy = xy[None, None]
    elif xy.ndim == 3:
        xy = xy[None]

    if mode == 'frame':
        mn = np.nanmin(xy, axis=2, keepdims=True)
        mx = np.nanmax(xy, axis=2, keepdims=True)
        return (xy - mn) / np.maximum(mx - mn, 1e-6) * 2 - 1
    elif mode == 'window':
        mn = np.nanmin(xy, axis=(1, 2), keepdims=True)      # (N,1,1,2)
        mx = np.nanmax(xy, axis=(1, 2), keepdims=True)
        center = (mn + mx) / 2
        half = np.maximum((mx - mn).max(axis=-1, keepdims=True) / 2, 1e-6)
        return (xy - center) / half
    raise ValueError(f"norm-mode must be 'frame' or 'window', got {mode}")


def prepare_features(features, frame_skip=2, norm_mode='frame'):
    """(N, T, V, 3) raw -> (N, T', V, 3) ready for the network."""
    x = np.array(features[:, ::frame_skip], dtype=np.float32, copy=True)
    x[..., :2] = processing_data(x[..., :2], norm_mode)
    return x


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def load_split(path, frame_skip=2, norm_mode='frame'):
    """Returns x (N,T',V,3), y one-hot (N,C), vids (N,) or None."""
    with open(path, 'rb') as f:
        feats, labels = pickle.load(f)
    x = prepare_features(np.asarray(feats), frame_skip, norm_mode)
    y = np.asarray(labels, dtype=np.float32)
    vpath = os.path.splitext(path)[0] + '_vids.npy'
    vids = np.load(vpath) if os.path.exists(vpath) else None
    return x, y, vids


def make_dataset(x, y):
    """(N,T,V,3) -> tensor (N,3,T,V), the layout ST-GCN expects."""
    xt = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
    return TensorDataset(xt, torch.from_numpy(y).float())


def get_motion(x):
    """Motion stream input: frame-to-frame xy difference. (N,3,T,V)->(N,2,T-1,V)"""
    return x[:, :2, 1:, :] - x[:, :2, :-1, :]


# --------------------------------------------------------------------------- #
#  Augmentation (training only, on a (N,3,T,V) tensor)
# --------------------------------------------------------------------------- #
def augment_batch(x, flip_idx=FLIP_IDX, flip_prob=0.5, noise_std=0.01, scale_range=(0.9, 1.1)):
    x = x.clone()
    n = x.shape[0]
    flip = torch.rand(n, device=x.device) < flip_prob
    if flip.any():
        xf = x[flip][..., flip_idx]          # swap left/right joints
        xf[:, 0] = -xf[:, 0]                 # mirror x
        x[flip] = xf
    s = torch.empty(n, 1, 1, 1, device=x.device).uniform_(*scale_range)
    x[:, :2] = x[:, :2] * s + torch.randn_like(x[:, :2]) * noise_std
    return x
