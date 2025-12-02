#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Defend NPZ (DIR) datasets in-place for WFlib DF attackers.

- Slices to seq_len (default 5000) BEFORE any ops.
- Normalizes by TRAIN per-position maxima (Mockingbird parity).
Outputs:
  out_root/(train|valid|test).npz  with X shape (N, seq_len), dtype preserved.
"""

import os, argparse, json, math
import numpy as np
import torch

# ----------------------------
# I/O helpers
# ----------------------------
def load_npz(path):
    d = np.load(path, mmap_mode="r")
    assert "X" in d and "y" in d, f"Expected keys X,y in {path}"
    X, y = d["X"], d["y"]
    assert X.ndim == 2, f"Expected X shape (N,L); got {X.shape}"
    y = y.astype(np.int64).reshape(-1)
    return X, y, X.dtype

def save_npz(path, X, y, dtype):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, X=X.astype(dtype, copy=False), y=y.astype(np.int64, copy=False))

def class_indices(y):
    idx = {}
    for i, lab in enumerate(y):
        idx.setdefault(int(lab), []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in idx.items()}

# ----------------------------
# Target selection
# ----------------------------
def pick_targets_closed_world(y, num_targets=8):
    N = len(y)
    all_idx = np.arange(N)
    by_class = class_indices(y)
    pools = []
    for i in range(N):
        same = set(by_class[int(y[i])].tolist())
        mask = np.ones(N, dtype=bool)
        if same:
            mask[list(same)] = False
        cand = all_idx[mask]
        k = min(num_targets, len(cand))
        choose = np.random.choice(cand, size=k, replace=False) if k > 0 else np.array([], dtype=np.int64)
        pools.append(choose)
    return pools

# ----------------------------
# Metrics
# ----------------------------
def rel_overhead(orig, adv):
    m = orig != 0
    if not np.any(m):
        return 0.0
    return float(np.sum(np.abs(adv[m] - orig[m])) / (np.sum(np.abs(orig[m])) + 1e-12))

# ----------------------------
# Additive, one-sided step (Mockingbird-like, in DIR)
# ----------------------------
def kmean_target(xi, T, kmean=4):
    # xi: (1,L), T: (K,L)  torch
    if T.shape[0] == 1:
        return T
    d2 = ((T - xi)**2).sum(dim=1)          # (K,)
    k = min(kmean, T.shape[0])
    idx = torch.topk(-d2, k).indices       # k nearest (largest -d2)
    return T[idx].mean(dim=0, keepdim=True)

def step_additive(x, t, alpha, step_cap=None, abs_cap_frac=None):
    """
    x,t: (1,L) in normalized space ~[0,1].
    One-sided additive toward target: x <- x + alpha_eff * ReLU(t - x)
    """
    gap_pos = torch.clamp(t - x, min=0.0)          # only grow where target > x
    dist = torch.linalg.norm(gap_pos)
    alpha_eff = alpha * float(dist)                # adaptive step: larger when far
    x_new = x + alpha_eff * gap_pos

    # Optional caps to bound overhead
    if step_cap is not None:
        x_new = torch.minimum(x + step_cap * t, x_new)
    if abs_cap_frac is not None:
        x_new = torch.minimum(x + abs_cap_frac * t, x_new)

    x_new = torch.maximum(x_new, x)                # one-sided
    return x_new, float(dist)

# ----------------------------
# White-box (DF logit push)
# ----------------------------
def build_df_model(num_classes):
    # Import lazily so users without WFlib can still run `add` strategy
    from WFlib import models
    model = models.DF(num_classes)
    return model

def step_logit(x, y_i, df_model, alpha, device):
    """
    x: (1,L) normalized; y_i: int
    One-sided additive step guided by ∂logit_c/∂x (padding-only).
    """
    x_in = x.view(1, 1, -1).detach().clone().requires_grad_(True)  # (1,1,L)
    logits = df_model(x_in)                                        # (1,C)
    zc = logits[0, int(y_i)]
    zc.backward()                                                  # ∂zc/∂x
    g = x_in.grad.detach().view(1, -1)                             # (1,L)

    # Move in -grad direction; keep positive parts only (additive)
    descent = -g
    pos = torch.clamp(descent, min=0.0)
    norm = torch.linalg.norm(pos) + 1e-12
    dir_vec = pos / norm

    alpha_eff = alpha * float(norm)
    x_new = x + alpha_eff * dir_vec
    x_new = torch.maximum(x_new, x)
    return x_new

# ----------------------------
# Core loops
# ----------------------------
def defend_split_additive(Xn, pools, iterations, alpha, patience, device,
                          kmean=4, step_cap=None, abs_cap_frac=None):
    dev = torch.device(device)
    X = torch.from_numpy(Xn).to(dev)    # (N,L)
    out = X.clone()
    N, L = X.shape

    for i in range(N):
        xi = X[i:i+1].detach().clone()
        pool_idx = pools[i]
        if pool_idx.size == 0:
            out[i:i+1] = xi
            continue
        T = X[pool_idx]                                       # (K,L)
        tgt = kmean_target(xi, T, kmean=kmean).detach()

        no_prog = 0
        last_dist = 1e9
        for _ in range(iterations):
            x = xi.clone()
            x_new, dist = step_additive(x, tgt, alpha, step_cap, abs_cap_frac)
            progressed = dist < last_dist - 1e-8
            no_prog = 0 if progressed else (no_prog + 1)
            xi, last_dist = x_new.detach(), dist

            # retarget if stuck
            if no_prog >= patience and T.shape[0] > 1:
                tgt = kmean_target(xi, T, kmean=kmean).detach()
                no_prog = 0

        out[i:i+1] = xi
    return out.cpu().numpy()

def defend_split_logit(Xn, y, iterations, alpha, device,
                       df_model):
    dev = torch.device(device)
    X = torch.from_numpy(Xn).to(dev)    # (N,L)
    out = X.clone()
    N, L = X.shape
    df_model.to(dev)
    df_model.eval()

    for i in range(N):
        xi = X[i:i+1].detach().clone()
        yi = int(y[i])
        for _ in range(iterations):
            xi = step_logit(xi, yi, df_model, alpha, dev)
        out[i:i+1] = xi.detach()
    return out.cpu().numpy()

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root",  default="./datasets/CW", help="Folder with train/valid/test npz")
    ap.add_argument("--out_root", default="./datasets/CW_MBDF", help="Output dataset folder")
    ap.add_argument("--mode", choices=["runtime","adaptive"], default="runtime",
                    help="runtime: defend valid/test only; adaptive: defend all splits")
    ap.add_argument("--strategy", choices=["add","logit"], default="add",
                    help="Defense strategy: 'add' (Mockingbird-like) or 'logit' (white-box DF)")
    ap.add_argument("--seq_len", type=int, default=5000)
    ap.add_argument("--num_targets", type=int, default=12)
    ap.add_argument("--iterations", type=int, default=180)
    ap.add_argument("--alpha", type=float, default=0.03)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--kmean", type=int, default=4, help="k for k-nearest mean target (add strategy)")
    ap.add_argument("--step_cap", type=float, default=0.05, help="per-step cap as fraction of target (add)")
    ap.add_argument("--abs_cap_frac", type=float, default=0.30, help="absolute cap as fraction of target (add)")
    ap.add_argument("--device", default="cpu")  # e.g., cuda:1

    # White-box args
    ap.add_argument("--ckpt_dataset", type=str, default="CW", help="Where DF checkpoint is saved")
    ap.add_argument("--load_name", type=str, default="max_f1", help="Checkpoint filename (without .pth)")

    args = ap.parse_args()
    os.makedirs(args.out_root, exist_ok=True)

    # Load splits
    trX, trY, tr_dtype = load_npz(os.path.join(args.in_root, "train.npz"))
    vaX, vaY, va_dtype = load_npz(os.path.join(args.in_root, "valid.npz"))
    teX, teY, te_dtype = load_npz(os.path.join(args.in_root, "test.npz"))

    # Slice to the part DF actually sees
    L = args.seq_len
    trX, vaX, teX = trX[:, :L], vaX[:, :L], teX[:, :L]

    # Normalize by TRAIN per-position maxima (1, L)
    vmax = np.maximum(trX.max(axis=0, keepdims=True), 1e-9)
    trXn, vaXn, teXn = trX / vmax, vaX / vmax, teX / vmax

    # Pools for 'add' strategy
    if args.strategy == "add":
        tr_pool = pick_targets_closed_world(trY, num_targets=args.num_targets)
        va_pool = pick_targets_closed_world(vaY, num_targets=args.num_targets)
        te_pool = pick_targets_closed_world(teY, num_targets=args.num_targets)

    # Optional DF for 'logit' strategy
    df_model = None
    if args.strategy == "logit":
        num_classes = int(max(trY.max(), vaY.max(), teY.max()) + 1)
        df_model = build_df_model(num_classes)
        # Load checkpoint from ./checkpoints/<ckpt_dataset>/DF/<load_name>.pth
        ckp = os.path.join("./checkpoints", args.ckpt_dataset, "DF", f"{args.load_name}.pth")
        if not os.path.exists(ckp):
            raise FileNotFoundError(f"DF checkpoint not found: {ckp}")
        state = torch.load(ckp, map_location="cpu")
        df_model.load_state_dict(state)

    # Defend
    if args.mode == "runtime":
        # train remains clean
        if args.strategy == "add":
            va_def = defend_split_additive(vaXn, va_pool, args.iterations, args.alpha,
                                           args.patience, args.device,
                                           kmean=args.kmean, step_cap=args.step_cap,
                                           abs_cap_frac=args.abs_cap_frac)
            te_def = defend_split_additive(teXn, te_pool, args.iterations, args.alpha,
                                           args.patience, args.device,
                                           kmean=args.kmean, step_cap=args.step_cap,
                                           abs_cap_frac=args.abs_cap_frac)
        else:
            va_def = defend_split_logit(vaXn, vaY, args.iterations, args.alpha, args.device, df_model)
            te_def = defend_split_logit(teXn, teY, args.iterations, args.alpha, args.device, df_model)

        va_def = (va_def * vmax).astype(va_dtype, copy=False)
        te_def = (te_def * vmax).astype(te_dtype, copy=False)

        save_npz(os.path.join(args.out_root, "train.npz"), trX, trY, tr_dtype)
        save_npz(os.path.join(args.out_root, "valid.npz"), va_def, vaY, va_dtype)
        save_npz(os.path.join(args.out_root, "test.npz"),  te_def, teY, te_dtype)

        # Overhead report
        print(f"[Overhead] valid: {100*rel_overhead(vaX, va_def):.2f}% | test: {100*rel_overhead(teX, te_def):.2f}%")

    else:  # adaptive
        if args.strategy == "add":
            tr_def = defend_split_additive(trXn, tr_pool, args.iterations, args.alpha,
                                           args.patience, args.device,
                                           kmean=args.kmean, step_cap=args.step_cap,
                                           abs_cap_frac=args.abs_cap_frac)
            va_def = defend_split_additive(vaXn, va_pool, args.iterations, args.alpha,
                                           args.patience, args.device,
                                           kmean=args.kmean, step_cap=args.step_cap,
                                           abs_cap_frac=args.abs_cap_frac)
            te_def = defend_split_additive(teXn, te_pool, args.iterations, args.alpha,
                                           args.patience, args.device,
                                           kmean=args.kmean, step_cap=args.step_cap,
                                           abs_cap_frac=args.abs_cap_frac)
        else:
            tr_def = defend_split_logit(trXn, trY, args.iterations, args.alpha, args.device, df_model)
            va_def = defend_split_logit(vaXn, vaY, args.iterations, args.alpha, args.device, df_model)
            te_def = defend_split_logit(teXn, teY, args.iterations, args.alpha, args.device, df_model)

        tr_def = (tr_def * vmax).astype(tr_dtype, copy=False)
        va_def = (va_def * vmax).astype(va_dtype, copy=False)
        te_def = (te_def * vmax).astype(te_dtype, copy=False)

        save_npz(os.path.join(args.out_root, "train.npz"), tr_def, trY, tr_dtype)
        save_npz(os.path.join(args.out_root, "valid.npz"), va_def, vaY, va_dtype)
        save_npz(os.path.join(args.out_root, "test.npz"),  te_def, teY, te_dtype)

        print(f"[Overhead] train: {100*rel_overhead(trX, tr_def):.2f}% | valid: {100*rel_overhead(vaX, va_def):.2f}% | test: {100*rel_overhead(teX, te_def)::.2f}%")

    # Manifest
    manifest = {
        "mode": args.mode,
        "strategy": args.strategy,
        "seq_len": args.seq_len,
        "num_targets": args.num_targets,
        "iterations": args.iterations,
        "alpha": args.alpha,
        "patience": args.patience,
        "kmean": args.kmean,
        "step_cap": args.step_cap,
        "abs_cap_frac": args.abs_cap_frac,
        "device": args.device,
    }
    with open(os.path.join(args.out_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

if __name__ == "__main__":
    main()
