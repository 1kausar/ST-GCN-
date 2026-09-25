"""
MediaPipe backend: Pose (33) + Hands (21 + 21) = 75 node.

    KeypointExtractor : ekta frame (BGR) -> (75, 3)   [x, y (0..1), conf]
    extract_sequence_mp : puro video -> (T, 75, 3)    (detect na hole NaN)

Node order (Utils.py-er Graph-er sathe hubohu mile):
    0-32  : MediaPipe Pose landmark
    33-53 : LEFT  hand (33 = wrist)
    54-74 : RIGHT hand (54 = wrist)

Left/right hand thik kora hoy pose-er wrist (15 = left, 16 = right)-er kache kon hand,
shei hishebe - MediaPipe Hands-er handedness label (mirror confusion) use kora hoy na.
Extraction, demo, show_pose - tinta jaygay ekoi class, tai training ar inference-er
data-te left/right kokhono ulte jabe na.

Note: mediapipe 'solutions' API lage -> pip install mediapipe==0.10.14
(notun version-e mp.solutions bad geche).
"""
import numpy as np
import cv2

from util.pose import iter_sampled_frames

N_NODE = 75
LEFT_WRIST, RIGHT_WRIST = 15, 16     # pose landmark index
HAND_L0, HAND_R0 = 33, 54            # hand block-er shuru


def _import_mediapipe():
    import mediapipe as mp
    if not hasattr(mp, 'solutions'):
        raise ImportError(
            "Tomar mediapipe version-e 'mp.solutions' nai. Install koro:\n"
            "    pip install 'mediapipe==0.10.14' 'numpy<2'")
    return mp


class KeypointExtractor:
    """frame (BGR) -> (75, 3). Detect na hole NaN. Video-r jonno tracking mode on."""

    def __init__(self, pose_conf=0.5, hand_conf=0.5, use_hands=True, model_complexity=1):
        mp = _import_mediapipe()
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False, model_complexity=model_complexity,
            min_detection_confidence=pose_conf, min_tracking_confidence=pose_conf)
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=1,
            min_detection_confidence=hand_conf, min_tracking_confidence=hand_conf) \
            if use_hands else None

    def close(self):
        self.pose.close()
        if self.hands is not None:
            self.hands.close()

    def __call__(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        kp = np.full((N_NODE, 3), np.nan, dtype=np.float32)

        pres = self.pose.process(rgb)
        if pres.pose_landmarks is None:
            return kp                                    # body nai -> puro frame NaN
        for i, lm in enumerate(pres.pose_landmarks.landmark):
            kp[i] = (lm.x, lm.y, lm.visibility)

        if self.hands is not None:
            hres = self.hands.process(rgb)
            if hres.multi_hand_landmarks:
                wl, wr = kp[LEFT_WRIST, :2], kp[RIGHT_WRIST, :2]
                for hl in hres.multi_hand_landmarks:
                    pts = np.array([(p.x, p.y) for p in hl.landmark], dtype=np.float32)
                    d_l = np.linalg.norm(pts[0] - wl)
                    d_r = np.linalg.norm(pts[0] - wr)
                    base = HAND_L0 if d_l <= d_r else HAND_R0
                    if not np.isnan(kp[base, 0]):        # oi side-e already ekta hand boshe gese
                        base = HAND_R0 if base == HAND_L0 else HAND_L0
                        if not np.isnan(kp[base, 0]):
                            continue
                    kp[base:base + 21, :2] = pts
                    kp[base:base + 21, 2] = 1.0
        return kp


def extract_sequence_mp(video_path, target_fps=30, pose_conf=0.5, hand_conf=0.5,
                        use_hands=True):
    """Puro video -> raw (T, 75, 3) float32. Frame sampling iter_sampled_frames-er moto
    (video fps > target_fps hole frame drop)."""
    ext = KeypointExtractor(pose_conf, hand_conf, use_hands)
    frames = []
    try:
        for _, frame in iter_sampled_frames(video_path, target_fps):
            frames.append(ext(frame))
    finally:
        ext.close()
    if not frames:
        return np.zeros((0, N_NODE, 3), dtype=np.float32)
    return np.stack(frames).astype(np.float32)
