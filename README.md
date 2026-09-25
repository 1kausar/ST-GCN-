# my_STGCN - Two-stream ST-GCN action recognition (MediaPipe body + hand, 75 node)

## Folder structure
```
my_STGCN/
├── config.yaml                  <- shob setting ekhane
├── train.py  test.py  demo.py  show_pose.py
├── data/
│   ├── extract_pose_mediapipe.py   STEP 1  video -> skeleton .npy
│   ├── create_dataset.py           STEP 2  .npy -> train/val/test .pkl
│   ├── videos/<class>/*.mp4        <- tomar video ekhane
│   └── poses/<class>/*.npy         (auto-generated)
├── models/   stgcn.py  Utils.py    (model + skeleton graph)
├── dataloader/dataset.py           (normalise, augment)
├── util/     pose.py  mp_pose.py  plot.py
└── runs/exp0/                      (training output)
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow
1. `config.yaml`-e `class-names` nijer class diye bodlao. Nam-gulo `data/videos/`-er sub-folder-er nam-er sathe mille jete hobe.
2. Video rakho: `data/videos/<class>/*.mp4` (prottek class-e kompokkhe 10+ video bhalo, minimum 3).
3. Pipeline:
```bash
python show_pose.py --video data/videos/class_a/x.mp4    # (optional) skeleton thik ashche kina dekho
python data/extract_pose_mediapipe.py                  # STEP 1
python data/create_dataset.py                            # STEP 2
python train.py                                          # STEP 3  -> runs/exp0/
python test.py --weights runs/exp1/best.pt               # STEP 4  (held-out test videos)
python demo.py --video test_td.mp4 --weights runs/exp1/best.pt   # STEP 5
```

## Notes
* Split video-level (ekta video-r window train ar test duto-te jay na) -> data leakage nai.
* `best.pt` validation accuracy diye select hoy, test set kokhono use hoy na.
* `test.py` window-level ar video-level (window-er mean) dutoi accuracy dey.
* MediaPipe Pose single-person; frame-e ekjon-i manush track hoy.
