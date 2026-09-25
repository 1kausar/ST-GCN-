"""
STEP 3 - Train two-stream ST-GCN.

    python train.py
    python train.py --config config.yaml

Output: runs/exp{n}/best.pt  last.pt  meta.json  history.json  result.png
(best.pt = VALIDATION accuracy sobcheye bhalo jei epoch-e; test set-e kokhono
 model select kora hoyni.)
"""
import os
import json
import time
import random
import argparse
import datetime

import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.stgcn import TwoStreamSpatialTemporalGraph
from dataloader.dataset import load_split, make_dataset, get_motion, augment_batch, FLIP_IDX


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, augment=False, flip_idx=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    pbar = tqdm(loader, leave=False, unit='batch', desc='train' if training else 'valid')
    with torch.set_grad_enabled(training):
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            if training and augment:
                x = augment_batch(x, flip_idx=flip_idx)
            mot = get_motion(x)
            out = model((x, mot))
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            bs = x.size(0)
            total_loss += loss.item() * bs
            correct += (out.argmax(1) == y.argmax(1)).sum().item()
            n += bs
            pbar.set_postfix(loss=f'{loss.item():.4f}')
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='config.yaml')
    args = ap.parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    class_names = cfg['class-names']
    num_class = len(class_names)

    layout = cfg.get('graph-layout', 'body_hand')   # MediaPipe 75-node (33 body + 21+21 hand)
    GRAPH_ARGS = {'strategy': cfg.get('graph-strategy', 'spatial'), 'layout': layout,
                 'max_hop': cfg.get('graph-max-hop', 3)}
    flip_idx = FLIP_IDX
    print(f'Device: {device} | classes: {class_names} | graph layout: {layout}')

    # ---------------- data ----------------
    x_tr, y_tr, _ = load_split(cfg['dataset-path-train'], cfg['frame-skip'], cfg['norm-mode'])
    x_va, y_va, _ = load_split(cfg['dataset-path-val'], cfg['frame-skip'], cfg['norm-mode'])
    assert y_tr.shape[1] == num_class, \
        f'pkl-e {y_tr.shape[1]} class, config-e {num_class} class - create_dataset abar chalao'
    for name, y in (('train', y_tr), ('val', y_va)):
        cnt = y.sum(0).astype(int).tolist()
        print(f'{name:<5}: {len(y)} windows | ' +
              ', '.join(f'{c}={k}' for c, k in zip(class_names, cnt)))

    pin = device.type == 'cuda'
    train_loader = DataLoader(make_dataset(x_tr, y_tr), batch_size=cfg['batch-size'],
                              shuffle=True, num_workers=cfg['num-workers'], pin_memory=pin)
    val_loader = DataLoader(make_dataset(x_va, y_va), batch_size=cfg['batch-size'],
                            shuffle=False, num_workers=cfg['num-workers'], pin_memory=pin)

    # ---------------- output folder ----------------
    os.makedirs(cfg['project'], exist_ok=True)
    k = 0
    while os.path.exists(os.path.join(cfg['project'], f'exp{k}')):
        k += 1
    save_dir = os.path.join(cfg['project'], f'exp{k}')
    os.makedirs(save_dir)
    meta = {'class_names': class_names, 'frame_skip': cfg['frame-skip'],
            'norm_mode': cfg['norm-mode'], 'num_frame': cfg['num-frame'],
            'graph_args': GRAPH_ARGS}
    with open(os.path.join(save_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(save_dir, 'config_used.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # ---------------- model ----------------
    model = TwoStreamSpatialTemporalGraph(GRAPH_ARGS, num_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = torch.nn.BCELoss()            # model output = sigmoid, labels = one-hot

    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc, best_loss, best_epoch = -1.0, float('inf'), -1
    for epoch in range(cfg['epochs']):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer,
                                    augment=cfg['augment'], flip_idx=flip_idx)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        for key, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc)):
            hist[key].append(v)

        improved = va_acc > best_acc or (va_acc == best_acc and va_loss < best_loss)
        if improved:
            best_acc, best_loss, best_epoch = va_acc, va_loss, epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pt'))
        print(f'Epoch {epoch + 1:>3}/{cfg["epochs"]} | '
              f'train loss {tr_loss:.4f} acc {tr_acc:.4f} | '
              f'val loss {va_loss:.4f} acc {va_acc:.4f} | '
              f'{str(datetime.timedelta(seconds=int(time.time() - t0)))}'
              f'{"  <- best" if improved else ""}')

    torch.save(model.state_dict(), os.path.join(save_dir, 'last.pt'))
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(hist, f, indent=2)

    # ---------------- curves ----------------
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist['train_acc'], label='Train')
    plt.plot(hist['val_acc'], label='Val')
    plt.xlabel('epoch'); plt.title('Accuracy'); plt.legend(loc='best')
    plt.subplot(1, 2, 2)
    plt.plot(hist['train_loss'], label='Train')
    plt.plot(hist['val_loss'], label='Val')
    plt.xlabel('epoch'); plt.title('Loss'); plt.legend(loc='best')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, 'result.png'), dpi=200)
    plt.close(fig)

    print(f'\nBest val acc {best_acc:.4f} @ epoch {best_epoch + 1}')
    print(f'Saved in: {save_dir}')
    print(f'Ebar: python test.py --weights {save_dir}/best.pt')


if __name__ == '__main__':
    main()
