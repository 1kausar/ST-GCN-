"""
STEP 4 - Evaluate a trained model on the (held-out) TEST videos.

    python test.py --weights runs/exp0/best.pt
    python test.py --weights runs/exp0/best.pt --split val

Output (runs/exp0/eval_test/):
    report.txt, metrics.json, confusion_matrix.png, confusion_matrix_normalize.png,
    video_level.csv   (window prediction gulo average kore video-wise result)
"""
import os
import json
import argparse

import yaml
import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader

from models.stgcn import TwoStreamSpatialTemporalGraph
from dataloader.dataset import load_split, make_dataset, get_motion
from util.plot import plot_cm


@torch.no_grad()
def predict(model, x, device, batch_size=64):
    loader = DataLoader(make_dataset(x, np.zeros((len(x), 1), np.float32)),
                        batch_size=batch_size, shuffle=False)
    probs = []
    model.eval()
    for xb, _ in loader:
        xb = xb.to(device)
        probs.append(model((xb, get_motion(xb))).cpu().numpy())
    return np.concatenate(probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--batch-size', type=int, default=64)
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    run_dir = os.path.dirname(os.path.abspath(args.weights))
    meta_path = os.path.join(run_dir, 'meta.json')
    if os.path.exists(meta_path):                    # training-time setting priority
        with open(meta_path) as f:
            meta = json.load(f)
    else:
        meta = {'class_names': cfg['class-names'], 'frame_skip': cfg['frame-skip'],
                'norm_mode': cfg['norm-mode'], 'graph_args': {'strategy': 'spatial'}}
    class_names = meta['class_names']
    C = len(class_names)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TwoStreamSpatialTemporalGraph(meta['graph_args'], C).to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))

    x, y, vids = load_split(cfg[f'dataset-path-{args.split}'], meta['frame_skip'], meta['norm_mode'])
    y_true = y.argmax(1)
    probs = predict(model, x, device, args.batch_size)
    y_pred = probs.argmax(1)
    labels = list(range(C))

    out_dir = os.path.join(run_dir, f'eval_{args.split}')
    os.makedirs(out_dir, exist_ok=True)

    # ---------------- window level ----------------
    acc = metrics.accuracy_score(y_true, y_pred)
    p, r, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0)
    report = metrics.classification_report(y_true, y_pred, labels=labels,
                                           target_names=class_names, digits=4, zero_division=0)
    cm = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    plot_cm(cm.T, normalize=False, save_dir=out_dir, names_x=class_names, names_y=class_names)
    plot_cm(cm.T, normalize=True, save_dir=out_dir, names_x=class_names, names_y=class_names)

    text = [f'Split: {args.split}  | windows: {len(y_true)}',
            f'Window-level accuracy : {acc * 100:.2f}%',
            f'Macro precision/recall/F1: {p:.4f} / {r:.4f} / {f1:.4f}', '', report]
    result = {'split': args.split, 'n_windows': int(len(y_true)), 'window_accuracy': acc,
              'macro_precision': p, 'macro_recall': r, 'macro_f1': f1,
              'confusion_matrix': cm.tolist(), 'class_names': class_names}

    # ---------------- video level ----------------
    if vids is not None and len(vids) == len(y_true):
        rows, v_true, v_pred = [], [], []
        for v in sorted(set(vids.tolist())):
            m = vids == v
            mean_p = probs[m].mean(0)
            t, pr = int(y_true[m][0]), int(mean_p.argmax())
            v_true.append(t)
            v_pred.append(pr)
            rows.append(f'{v},{class_names[t]},{class_names[pr]},{mean_p[pr]:.4f},{int(t == pr)}')
        v_acc = metrics.accuracy_score(v_true, v_pred)
        v_f1 = metrics.f1_score(v_true, v_pred, labels=labels, average='macro', zero_division=0)
        with open(os.path.join(out_dir, 'video_level.csv'), 'w') as f:
            f.write('video,true,pred,confidence,correct\n' + '\n'.join(rows) + '\n')
        text += ['', f'Video-level accuracy (mean of window probs): {v_acc * 100:.2f}%  '
                     f'({len(v_true)} videos), macro-F1 {v_f1:.4f}']
        result.update({'n_videos': len(v_true), 'video_accuracy': v_acc, 'video_macro_f1': v_f1})

    text = '\n'.join(text)
    print(text)
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved in: {out_dir}')


if __name__ == '__main__':
    main()
