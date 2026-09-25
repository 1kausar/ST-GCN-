"""
Video chola obosthay body + hand keypoint (skeleton) live dekhar script.
Kono trained model lage na - shudhu MediaPipe Pose (33) + Hands (21+21) = 75 node.

    python show_pose.py --video /home/kausar/Downloads/test_video.mp4
    python show_pose.py --video test_video.mp4 --save out.mp4     # dekhar sathe video-o save
    python show_pose.py --video test_video.mp4 --ids              # joint number (0-74)
    python show_pose.py --video test_video.mp4 --no-hands         # shudhu body (aro fast)
    python show_pose.py --video 0                                 # webcam

Key:  q / Esc = bondho   |   space = pause / resume
Window khola na gele (display nai) automatic video file-e save hobe.
Note: MediaPipe Pose single-person - frame-e sobcheye clear manush-take track kore.
"""
import os
import sys
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import yaml
import numpy as np

from models.Utils import Graph
from util.mp_pose import KeypointExtractor, N_NODE, HAND_L0, HAND_R0

BLUE = (255, 0, 0)        # BGR: body bone
ORANGE = (0, 165, 255)    # BGR: hand bone
GREEN = (0, 255, 0)       # BGR: joint point


def draw(frame, kpts, edges, thr, show_ids=False):
    """kpts: (75,3) x,y normalised (0..1) + confidence."""
    h, w = frame.shape[:2]
    th = max(2, int(min(h, w) / 200))
    ok = ~np.isnan(kpts[:, 0]) & (np.nan_to_num(kpts[:, 2]) >= thr)
    pts = np.nan_to_num(kpts[:, :2]) * [w, h]
    pts = pts.astype(np.int32)
    for a, b in edges:
        if ok[a] and ok[b]:
            col = ORANGE if (a >= 33 or b >= 33) else BLUE
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), col, th, cv2.LINE_AA)
    for i in range(N_NODE):
        if ok[i]:
            r = th + 1 if i < 33 else max(1, th - 1)      # hand point ektu chhoto
            cv2.circle(frame, tuple(pts[i]), r, GREEN, -1, cv2.LINE_AA)
            if show_ids:
                cv2.putText(frame, str(i), (pts[i][0] + 4, pts[i][1] - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)


def open_writer(path, fps, w, h):
    print(f'Video save hocche: {path}')
    return cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help="video path, ba webcam-er jonno 0")
    ap.add_argument('--config', default='config.yaml')
    ap.add_argument('--save', default=None, help='output video path (optional)')
    ap.add_argument('--ids', action='store_true', help='joint-er number (0-74) dekhao')
    ap.add_argument('--no-hands', action='store_true', help='hand detect korbe na (shudhu body)')
    ap.add_argument('--no-show', action='store_true', help='window na khule shudhu file-e save')
    args = ap.parse_args()

    cfg = {}
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
    pose_conf = cfg.get('pose-conf', 0.5)
    hand_conf = cfg.get('hand-conf', pose_conf)
    kpt_thr = cfg.get('kpt-conf', 0.2)

    extractor = KeypointExtractor(pose_conf, hand_conf, use_hands=not args.no_hands)
    edges = [(a, b) for a, b in Graph(layout='body_hand').edge if a != b]

    src = int(args.video) if args.video.isdigit() else args.video
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f'Video can not open: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps == fps and fps >= 1 else 30.0

    default_out = (os.path.splitext(args.video)[0] if not args.video.isdigit() else 'webcam') + '_pose.mp4'
    out_path = args.save or default_out
    show = not args.no_show
    writer = None
    win = 'Body + hand keypoints  (q = quit, space = pause)'
    paused, n, fps_smooth = False, 0, 0.0
    frame = None

    while True:
        t0 = time.time()
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            kp = extractor(frame)
            body_ok = not np.isnan(kp[0, 0])
            draw(frame, kp, edges, kpt_thr, args.ids)
            n_hands = int(sum(not np.isnan(kp[b, 0]) for b in (HAND_L0, HAND_R0)))
            dt = max(time.time() - t0, 1e-6)
            fps_smooth = 1 / dt if n == 0 else 0.9 * fps_smooth + 0.1 / dt
            cv2.putText(frame, f'frame {n}  body {"yes" if body_ok else "no"}  hands {n_hands}  '
                               f'{fps_smooth:.1f} FPS', (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
            if writer is None and (args.save or not show):
                h, w = frame.shape[:2]
                writer = open_writer(out_path, fps, w, h)
            if writer is not None:
                writer.write(frame)
            n += 1

        if show:
            try:
                cv2.imshow(win, frame)
                delay = 30 if paused else max(1, int(1000 / fps - (time.time() - t0) * 1000))
                key = cv2.waitKey(delay) & 0xFF
            except cv2.error:
                print('Window kholte parlam na (display / opencv GUI nai).')
                show = False
                if writer is None and frame is not None:
                    h, w = frame.shape[:2]
                    writer = open_writer(out_path, fps, w, h)
                continue
            if key in (ord('q'), 27):
                break
            if key == 32:
                paused = not paused

    cap.release()
    extractor.close()
    if writer is not None:
        writer.release()
        print(f'Saved: {out_path}')
    if show:
        cv2.destroyAllWindows()
    print(f'{n} frame processed.')


if __name__ == '__main__':
    main()
