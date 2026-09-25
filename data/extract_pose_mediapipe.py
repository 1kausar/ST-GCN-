"""
STEP 1 (MediaPipe backend) - .mp4 video -> body+hand skeleton (.npy)

Video-folder structure:  <video-dir>/<class>/*.mp4
Output: <pose-dir>/<class>/*.npy   (shape T x 75 x 3)
Prottek frame-e MediaPipe Pose (33 point) + duita hand (21+21 point) = 75 node.

Run (project root theke):
    python data/extract_pose_mediapipe.py
    python data/extract_pose_mediapipe.py --overwrite

config.yaml-e obossho eta thakte hobe:
    graph-layout: 'body_hand'
"""
import os
import sys
import csv
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import numpy as np
from tqdm import tqdm

from util.pose import clean_sequence
from util.mp_pose import extract_sequence_mp

VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.webm')


def find_videos(video_dir, class_names):
    """Returns list of (video_path, class_name) from <video_dir>/<class>/*.<ext>"""
    out = []
    for cname in class_names:
        cdir = os.path.join(video_dir, cname)
        if not os.path.isdir(cdir):
            print(f'[warn] folder nai: {cdir}')
            continue
        files = sorted(p for p in glob.glob(os.path.join(cdir, '*'))
                       if p.lower().endswith(VIDEO_EXTS))
        if not files:
            print(f'[warn] "{cname}" folder-e kono video nai')
        out += [(p, cname) for p in files]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if cfg.get('graph-layout', 'coco') != 'body_hand':
        print("Note: config.yaml-e 'graph-layout: body_hand' set kora nai. "
              "Extraction cholbe, kintu train.py-o eki graph-layout use korte hobe.")

    class_names = cfg['class-names']
    video_dir, pose_dir = cfg['video-dir'], cfg['pose-dir']
    videos = find_videos(video_dir, class_names)
    if not videos:
        sys.exit(f'Kono video pawa jayni: {video_dir}/<class>/*.mp4 structure check koro')
    print(f'{len(videos)} video pawa gelo. Backend: MediaPipe (Pose + Hands, 75 node)')

    os.makedirs(pose_dir, exist_ok=True)
    used_names, rows, skipped = set(), [], []

    for path, cname in tqdm(videos, unit='video'):
        stem = os.path.splitext(os.path.basename(path))[0]
        name = stem
        k = 1
        while (cname, name) in used_names:
            k += 1
            name = f'{stem}_{k}'
        used_names.add((cname, name))

        out_dir = os.path.join(pose_dir, cname)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, name + '.npy')
        if os.path.exists(out_path) and not args.overwrite:
            continue

        try:
            raw = extract_sequence_mp(path, target_fps=cfg['target-fps'],
                                      pose_conf=cfg['pose-conf'],
                                      hand_conf=cfg.get('hand-conf', cfg['pose-conf']))
        except Exception as e:
            print(f'[error] {path}: {e}')
            skipped.append(path)
            continue

        detected = float((~np.isnan(raw[:, 0, 0])).mean()) if len(raw) else 0.0
        seq = clean_sequence(raw, cfg['kpt-conf'], cfg['min-valid-ratio'])
        if seq is None:
            print(f'[skip] {path}  (body detect hoyeche shudhu {detected * 100:.0f}% frame-e)')
            skipped.append(path)
            continue

        np.save(out_path, seq)
        rows.append([cname, name, len(seq), f'{detected:.2f}'])

    if rows:
        summ = os.path.join(pose_dir, 'summary.csv')
        new = not os.path.exists(summ)
        with open(summ, 'a', newline='') as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(['class', 'video', 'frames_used', 'body_detected_ratio'])
            wr.writerows(rows)

    print(f'\nDone. {len(rows)} video processed, {len(skipped)} skipped.')
    if skipped:
        print('Skipped list:')
        for s in skipped:
            print('  ', s)
    print(f'Output: {pose_dir}/<class>/*.npy  (shape T x 75 x 3)  -> ebar: python data/create_dataset.py')


if __name__ == '__main__':
    main()
