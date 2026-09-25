"""
STEP 2 - skeleton sequences  ->  train / val / test  .pkl

* Split VIDEO-level e hoy (ekta video-r window train ar test duto-te jabe na).
  Original repo random window split korto - tate neighbouring window duto
  train/test-e pore jay = data leakage, accuracy fake beshi ashe.
* Prottek video theke sliding window (num-frame, window-stride) neya hoy.
* Label = one-hot (BCELoss-er jonno), video-r class.

Run:
    python data/create_dataset.py
"""
import os
import sys
import json
import glob
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np

from util.pose import sliding_windows


def split_counts(n, ratios):
    """n video -> (n_train, n_val, n_test). Non-zero ratio hole minimum 1."""
    r_tr, r_va, r_te = ratios
    n_va = max(1, int(round(n * r_va))) if r_va > 0 else 0
    n_te = max(1, int(round(n * r_te))) if r_te > 0 else 0
    n_tr = n - n_va - n_te
    return n_tr, n_va, n_te


def build(items, n_frames, stride, n_classes):
    xs, ys, vids = [], [], []
    for cls_idx, vid_name, seq in items:
        wins, _ = sliding_windows(seq, n_frames, stride)
        onehot = np.zeros((len(wins), n_classes), dtype=np.float32)
        onehot[:, cls_idx] = 1.0
        xs.append(wins)
        ys.append(onehot)
        vids += [vid_name] * len(wins)
    if not xs:
        return (np.zeros((0, n_frames, 17, 3), np.float32),
                np.zeros((0, n_classes), np.float32), np.array([], dtype='<U1'))
    return np.concatenate(xs), np.concatenate(ys), np.array(vids)


def save(path, x, y, vids):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump((x, y), f)                          # same format as original repo
    np.save(os.path.splitext(path)[0] + '_vids.npy', vids)  # video id of each window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    class_names = cfg['class-names']
    C = len(class_names)
    n_frames, stride = cfg['num-frame'], cfg['window-stride']
    ratios = cfg['split-ratio']
    assert abs(sum(ratios) - 1.0) < 1e-6, 'split-ratio sum 1 hote hobe'
    rng = np.random.RandomState(cfg['seed'])

    splits = {'train': [], 'val': [], 'test': []}
    split_info = {'class_names': class_names, 'num_frame': n_frames,
                  'window_stride': stride, 'seed': cfg['seed'],
                  'train': {}, 'val': {}, 'test': {}}

    for ci, cname in enumerate(class_names):
        files = sorted(glob.glob(os.path.join(cfg['pose-dir'], cname, '*.npy')))
        n = len(files)
        n_tr, n_va, n_te = split_counts(n, ratios)
        if n_tr < 1:
            sys.exit(f'Class "{cname}"-e video mane {n}-ta; train/val/test split korte '
                     f'kompokkhe 3 ta video lagbe (ideally 10+).')
        order = rng.permutation(n)
        parts = {'train': order[:n_tr],
                 'val': order[n_tr:n_tr + n_va],
                 'test': order[n_tr + n_va:]}
        for sp, idxs in parts.items():
            names = []
            for i in idxs:
                seq = np.load(files[i])
                vname = f'{cname}/{os.path.splitext(os.path.basename(files[i]))[0]}'
                splits[sp].append((ci, vname, seq))
                names.append(vname)
            split_info[sp][cname] = names

    # ---- build + save ----
    print(f'{"split":<7}' + ''.join(f'{c:>14}' for c in class_names) + f'{"total":>10}')
    for sp in ('train', 'val', 'test'):
        x, y, vids = build(splits[sp], n_frames, stride, C)
        save(cfg[f'dataset-path-{sp}'], x, y, vids)
        n_vid = [len(split_info[sp][c]) for c in class_names]
        n_win = y.sum(0).astype(int).tolist() if len(y) else [0] * C
        print(f'{sp:<7}' + ''.join(f'{f"{v}v/{w}w":>14}' for v, w in zip(n_vid, n_win)) +
              f'{len(x):>10}')
        print(f'        x shape = {x.shape}')

    info_path = os.path.join(os.path.dirname(cfg['dataset-path-train']) or '.', 'split_info.json')
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f'\n(v = video, w = window)  split list: {info_path}')
    print('Ebar: python train.py')


if __name__ == '__main__':
    main()
