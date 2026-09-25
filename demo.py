"""
STEP 5 (optional) - Naya .mp4 video-te predict kore skeleton + label draw kora video save.
(MediaPipe Pose + Hands, 75 node)

    python demo.py --video test.mp4 --weights runs/exp0/best.pt
    python demo.py --video test.mp4 --weights runs/exp0/best.pt --out result.mp4
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import yaml
import numpy as np
import torch

from models.stgcn import TwoStreamSpatialTemporalGraph
from models.Utils import Graph
from dataloader.dataset import prepare_features, get_motion
from util.pose import (clean_sequence, sliding_windows,
                       iter_sampled_frames, get_video_info)
from util.mp_pose import extract_sequence_mp


def make_edges(layout='body_hand'):
    g = Graph(layout=layout)
    return [(a, b) for a, b in g.edge if a != b]


def draw_skeleton(frame, kp, edges, thr=0.3):
    """kp: (75,3) normalised x,y (0..1) + conf. Body = green, hand = orange."""
    h, w = frame.shape[:2]
    ok = np.nan_to_num(kp[:, 2]) >= thr
    pts = (kp[:, :2] * [w, h]).astype(np.int32)
    for a, b in edges:
        if ok[a] and ok[b]:
            col = (0, 165, 255) if (a >= 33 or b >= 33) else (0, 255, 0)
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), col, 2, cv2.LINE_AA)
    for i in range(len(kp)):
        if ok[i]:
            cv2.circle(frame, tuple(pts[i]), 3, (0, 0, 255), -1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--out', default=None)
    ap.add_argument('--stride', type=int, default=5, help='window step (frames)')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    run_dir = os.path.dirname(os.path.abspath(args.weights))
    with open(os.path.join(run_dir, 'meta.json')) as f:
        meta = json.load(f)
    class_names, n_frames = meta['class_names'], meta['num_frame']
    C = len(class_names)

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    stgcn = TwoStreamSpatialTemporalGraph(meta['graph_args'], C).to(dev)
    stgcn.load_state_dict(torch.load(args.weights, map_location=dev))
    stgcn.eval()

    # ---- 1) pose for the whole video (MediaPipe) ----
    print('Pose extract......')
    raw = extract_sequence_mp(args.video, target_fps=cfg['target-fps'],
                              pose_conf=cfg['pose-conf'],
                              hand_conf=cfg.get('hand-conf', cfg['pose-conf']))
    seq = clean_sequence(raw, cfg['kpt-conf'], cfg['min-valid-ratio'])
    if seq is None:
        raise SystemExit('Video-te person detect hoyni / onek kom frame-e detect hoyeche.')
    T = len(seq)

    # ---- 2) sliding-window prediction ----
    wins, starts = sliding_windows(seq, n_frames, args.stride)
    x = torch.from_numpy(prepare_features(wins, meta['frame_skip'], meta['norm_mode']))
    x = x.permute(0, 3, 1, 2).contiguous().to(dev)
    with torch.no_grad():
        probs = stgcn((x, get_motion(x))).cpu().numpy()          # (Nw, C)

    frame_probs = np.zeros((T, C))
    frame_cnt = np.zeros(T)
    for s, p in zip(starts, probs):
        frame_probs[s:s + n_frames] += p
        frame_cnt[s:s + n_frames] += 1
    frame_probs = frame_probs / np.maximum(frame_cnt, 1)[:, None]

    video_probs = probs.mean(0)
    print('\nVideo-level prediction:')
    for c, p in sorted(zip(class_names, video_probs), key=lambda z: -z[1]):
        print(f'  {c:<12} {p * 100:6.2f}%')

    # ---- 3) render ----
    out_path = args.out or os.path.splitext(args.video)[0] + '_pred.mp4'
    fps, w, h = get_video_info(args.video)
    out_fps = min(fps, cfg['target-fps']) if cfg['target-fps'] else fps
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), out_fps, (w, h))
    edges = make_edges(meta['graph_args'].get('layout', 'body_hand'))
    top = class_names[int(video_probs.argmax())]
    for t, (_, frame) in enumerate(iter_sampled_frames(args.video, cfg['target-fps'])):
        if t >= T:
            break
        draw_skeleton(frame, seq[t], edges, cfg['kpt-conf'])
        k = int(frame_probs[t].argmax())
        cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.putText(frame, f'{class_names[k]}  {frame_probs[t][k] * 100:.0f}%', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f'video: {top} {video_probs.max() * 100:.0f}%', (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
