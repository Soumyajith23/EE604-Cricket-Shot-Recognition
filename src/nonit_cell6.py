# ===== CELL 1: paste this whole cell, run it (writes the two modules; prints sizes) =====
fe = r'''"""Stage 2: frozen EfficientNetV2-S features from the Stage-1 segmentation-overlay crops.

For every ok clip we keep ONLY the frames where the striker was tracked (the paper also drops striker-less frames)
and store, per frame:  2x2-avg-pooled last feature map (1280*2*2 = 5120, float16) for the crop AND for its
horizontally mirrored copy (needed for mirror augmentation / test-time augmentation), plus the pose / box arrays
compacted to the same frames.
"""
import glob
import os
import pickle
import time

import cv2
import numpy as np

FEAT_DIM = 5120


def find_cache(root="/kaggle/input"):
    """Locate the Stage-1 cache directory (the one with the most chunk_*.pkl files) under `root`."""
    best, bestn = None, 0
    for dp, _, fs in os.walk(root):
        n = sum(f.startswith("chunk_") and f.endswith(".pkl") for f in fs)
        if n > bestn:
            best, bestn = dp, n
    return best, bestn


def compact(rec):
    """Keep only striker-tracked frames."""
    v = np.where(rec["valid"])[0]
    return dict(idx=v.astype(np.int16), kp=rec["kp"][v], kc=rec["kc"][v], sbox=rec["sbox"][v],
                bat=rec["bat"][v], W=rec["W"], H=rec["H"], n_total=rec["n"])


def build_model(device):
    import torch
    import torchvision
    m = torchvision.models.efficientnet_v2_s(weights=torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    body = m.features.eval().to(device).half()
    for p in body.parameters():
        p.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1).half()
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1).half()
    return body, mean, std


def clip_features(jpgs, body, mean, std, device, min_batch=1):
    """jpgs: list of JPEG bytes (BGR overlay crops, 224x224). Returns (f, f_flip) as float16 arrays (n, 5120)."""
    import torch
    import torch.nn.functional as F
    ims = [cv2.cvtColor(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for j in jpgs]
    x = torch.from_numpy(np.stack(ims)).to(device).permute(0, 3, 1, 2).half().div_(255)
    x = (x - mean) / std
    x = torch.cat([x, x.flip(3)], 0)  # [originals ; mirrored]
    with torch.no_grad():
        fm = body(x)                                  # (2n, 1280, 7, 7)
        fm = F.adaptive_avg_pool2d(fm, 2).flatten(1)  # (2n, 5120)
    fm = fm.float().cpu().numpy().astype(np.float16)
    n = len(jpgs)
    return fm[:n], fm[n:]


def extract(cache_dir, out_dir, device="cuda", limit_chunks=0, log=print):
    os.makedirs(out_dir, exist_ok=True)
    body, mean, std = build_model(device)
    files = sorted(glob.glob(os.path.join(cache_dir, "chunk_*.pkl")))
    if limit_chunks:
        files = files[:limit_chunks]
    t0, nclips = time.time(), 0
    for fi, f in enumerate(files):
        out = os.path.join(out_dir, "feat_" + os.path.basename(f)[len("chunk_"):])
        if os.path.exists(out):
            continue
        recs = pickle.load(open(f, "rb"))
        res = []
        for r in recs:
            if not r.get("ok"):
                continue
            c = compact(r)
            if len(c["idx"]) < 4:
                continue
            jp = [r["jpg_seg"][i] for i in c["idx"]]
            c["f"], c["ff"] = clip_features(jp, body, mean, std, device)
            c.update(cls=r["cls"], match=r["match"], clip=r["clip"], path=r["path"])
            res.append(c)
            nclips += 1
        tmp = out + ".tmp"
        with open(tmp, "wb") as fh:
            pickle.dump(res, fh, protocol=4)
        os.replace(tmp, out)
        log(f"[{time.strftime('%H:%M:%S')}] {os.path.basename(out)} ({fi + 1}/{len(files)}) | {len(res)}/{len(recs)} clips kept"
            f" | {(time.time() - t0) / max(nclips, 1):.3f}s/clip")
    return nclips


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit_chunks", type=int, default=0)
    a = ap.parse_args()
    extract(a.cache, a.out, limit_chunks=a.limit_chunks)
'''
fu = r'''"""Stage 3: pose features, mirroring, phase alignment, match-level splits, RGB / pose / fusion training, evaluation.

Everything that does not need torch is plain numpy (so it can be unit-tested anywhere).
"""
import glob
import json
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False

FEAT_DIM = 5120
K_FRAMES = 15
# COCO-17: 0 nose, 1/2 eyes, 3/4 ears, 5/6 shoulders, 7/8 elbows, 9/10 wrists, 11/12 hips, 13/14 knees, 15/16 ankles
FLIP_PERM = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
ANGLE_TRIPLES = [(5, 7, 9), (6, 8, 10), (7, 5, 11), (8, 6, 12), (5, 11, 13), (6, 12, 14), (11, 13, 15), (12, 14, 16)]


# ----------------------------------------------------------------------------- pose features
def flip_pose(kp, kc, W):
    """Mirror a skeleton horizontally: x -> W-x and swap left/right joints."""
    k2 = kp[:, FLIP_PERM].copy()
    k2[..., 0] = W - k2[..., 0]
    return k2, kc[:, FLIP_PERM].copy()


def fill_missing(kp, kc, thr=0.25):
    """Low-confidence joints are replaced by linear interpolation over the (valid-frame) sequence."""
    n = len(kp)
    t = np.arange(n)
    out = kp.astype(np.float32).copy()
    good = kc >= thr
    cen = np.array([kp[i][good[i]].mean(0) if good[i].any() else kp[i].mean(0) for i in range(n)], np.float32)
    for j in range(kp.shape[1]):
        g = good[:, j]
        if g.sum() >= 1:
            for d in range(2):
                out[:, j, d] = np.interp(t, t[g], kp[g, j, d])
        else:
            out[:, j] = cen
    return out


def _angle(a, b, c):
    u, v = a - b, c - b
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-6)
    return np.arccos(np.clip(cos, -1, 1)) / np.pi


def pose_features(kp, kc, idx):
    """kp (n,17,2) image coords, kc (n,17), idx (n,) original frame numbers.
    Returns feats (n,68) and wrist speed (n,) in torso-lengths / frame.
    68 = 34 hip-centred torso-scaled coords + 17 confidences + 8 joint angles + 1 torso lean
         + 4 wrist velocities + 2 hip velocity + 2 hip displacement from first frame."""
    n = len(kp)
    k = fill_missing(kp, kc)
    hip = (k[:, 11] + k[:, 12]) / 2
    sho = (k[:, 5] + k[:, 6]) / 2
    sc = max(float(np.median(np.linalg.norm(sho - hip, axis=1))), 8.0)  # clip-level torso length in px
    rel = (k - hip[:, None]) / sc
    gap = np.maximum(np.diff(idx.astype(np.float32)), 1.0)

    def vel(x):
        v = np.zeros_like(x)
        if n > 1:
            v[1:] = np.diff(x, axis=0) / gap.reshape((-1,) + (1,) * (x.ndim - 1))
            v[0] = v[1]
        return v

    wv = vel(k[:, [9, 10]] / sc).reshape(n, 4)
    hv = vel(hip / sc)
    hd = (hip - hip[0]) / sc
    ang = np.stack([_angle(k[:, a], k[:, b], k[:, c]) for a, b, c in ANGLE_TRIPLES], 1)
    lean = (np.arctan2(sho[:, 0] - hip[:, 0], -(sho[:, 1] - hip[:, 1])) / np.pi)[:, None]
    feats = np.concatenate([rel.reshape(n, 34), kc, ang, lean, wv, hv, hd], 1).astype(np.float32)
    speed = np.maximum(np.linalg.norm(wv[:, :2], axis=1), np.linalg.norm(wv[:, 2:], axis=1))
    return feats, speed


def peak_index(speed, lo_frac=0.2, hi_frac=0.85):
    """Impact proxy = frame of maximum (smoothed) wrist speed, searched in the middle of the clip."""
    n = len(speed)
    s = np.convolve(np.pad(speed, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
    lo = int(lo_frac * (n - 1))
    hi = max(lo + 1, int(hi_frac * (n - 1)))
    return int(lo + np.argmax(s[lo:hi + 1]))


# ----------------------------------------------------------------------------- temporal sampling
def positions(nv, K=K_FRAMES, peak=None, jitter=False, rng=None, peak_k=8):
    """Indices (into the nv striker-valid frames) of the K frames fed to the network.
    peak=None  -> uniform sampling (paper).  peak=p -> piecewise-linear time warp so frame p lands on slot peak_k."""
    k = np.arange(K, dtype=np.float64)
    if jitter:
        k = np.sort(np.clip(k + rng.uniform(-0.5, 0.5, K), 0, K - 1))
    if peak is None or nv < 3:
        pos = k / (K - 1) * (nv - 1)
    else:
        pos = np.where(k <= peak_k, peak * k / peak_k, peak + (nv - 1 - peak) * (k - peak_k) / (K - 1 - peak_k))
    return np.clip(np.rint(pos).astype(int), 0, nv - 1)


# ----------------------------------------------------------------------------- data
def load_clips(feat_dir, classes=None, limit=0, log=print):
    """Load feature chunks -> list of per-clip dicts with RGB feats (orig+mirror) and pose feats (orig+mirror)."""
    files = sorted(glob.glob(os.path.join(feat_dir, "feat_*.pkl")))
    raw = []
    for f in files:
        raw += pickle.load(open(f, "rb"))
        if limit and len(raw) >= limit:
            raw = raw[:limit]
            break
    if classes is None:
        classes = sorted({r["cls"] for r in raw})
    cid = {c: i for i, c in enumerate(classes)}
    data = []
    for r in raw:
        kp, kc, idx, W = r["kp"], r["kc"], r["idx"], r["W"]
        P, sp = pose_features(kp, kc, idx)
        kpf, kcf = flip_pose(kp, kc, W)
        Pf, _ = pose_features(kpf, kcf, idx)
        data.append(dict(R=r["f"], Rf=r["ff"], P=P, Pf=Pf, nv=len(idx), peak=peak_index(sp),
                         y=cid[r["cls"]], match=int(r["match"]), cls=r["cls"]))
    log(f"loaded {len(data)} clips | {len(classes)} classes | mean valid frames {np.mean([d['nv'] for d in data]):.1f}")
    return data, classes


def make_split(matches, y, n_classes, seed, fr=(0.70, 0.15), tries=400):
    """Match-level split (no match appears in two splits). Among `tries` random match orders keep the one whose
    per-class proportions are closest to the target. Returns split id per clip (0 train / 1 val / 2 test)."""
    matches, y = np.asarray(matches), np.asarray(y)
    um, inv = np.unique(matches, return_inverse=True)
    M = np.zeros((len(um), n_classes))
    np.add.at(M, (inv, y), 1)
    tot = M.sum(0)
    target = np.stack([tot * fr[0], tot * fr[1], tot * (1 - fr[0] - fr[1])])
    rng = np.random.default_rng(seed)
    best, bscore = None, 1e18
    for _ in range(tries):
        perm = rng.permutation(len(um))
        cum = np.cumsum(M.sum(1)[perm]) / M.sum()
        a = np.empty(len(um), int)
        a[perm] = np.where(cum <= fr[0], 0, np.where(cum <= fr[0] + fr[1], 1, 2))
        cnt = np.stack([M[a == s].sum(0) for s in range(3)])
        score = (((cnt - target) / np.maximum(target, 1)) ** 2).sum()
        if score < bscore:
            best, bscore = a.copy(), score
    return best[inv]


# ----------------------------------------------------------------------------- metrics
def confusion(y, p, C):
    cm = np.zeros((C, C), int)
    np.add.at(cm, (y, p), 1)
    return cm


def per_class_f1(cm):
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    return np.where(tp > 0, tp / (tp + 0.5 * (fp + fn) + 1e-12), 0.0)


def metrics(y, probs, C):
    p = probs.argmax(1)
    cm = confusion(y, p, C)
    f1 = per_class_f1(cm)
    sup = cm.sum(1) > 0
    return dict(acc=float((p == y).mean()), f1=float(f1[sup].mean()), per_class=f1.tolist(), cm=cm.tolist())


# ----------------------------------------------------------------------------- batching
@dataclass
class Cfg:
    name: str
    rgb: bool = True
    pose: bool = False
    mirror: bool = False   # random mirroring in training + mirror TTA at evaluation
    align: bool = False    # phase-aligned temporal sampling
    moddrop: float = 0.0   # modality dropout probability (each modality), fusion only
    K: int = K_FRAMES


CFGS = [
    Cfg("rgb"),
    Cfg("rgb_m", mirror=True),
    Cfg("pose_m", rgb=False, pose=True, mirror=True),
    Cfg("pose_m_align", rgb=False, pose=True, mirror=True, align=True),
    Cfg("fuse", pose=True, moddrop=0.15),
    Cfg("fuse_m", pose=True, mirror=True, moddrop=0.15),
    Cfg("fuse_m_align", pose=True, mirror=True, align=True, moddrop=0.15),
]


def make_batch(data, ids, cfg, rng=None, train=False, flip=None):
    """flip=None: random per sample when training with cfg.mirror, else no flip. flip=True/False: forced (TTA)."""
    B, K = len(ids), cfg.K
    D = data[0]["P"].shape[1]
    xr = np.empty((B, K, FEAT_DIM), np.float16) if cfg.rgb else None
    xp = np.empty((B, K, D), np.float32) if cfg.pose else None
    for b, i in enumerate(ids):
        c = data[i]
        fl = bool(flip) if flip is not None else bool(train and cfg.mirror and rng.random() < 0.5)
        pos = positions(c["nv"], K, c["peak"] if cfg.align else None, jitter=train, rng=rng)
        if xr is not None:
            xr[b] = (c["Rf"] if fl else c["R"])[pos]
        if xp is not None:
            xp[b] = (c["Pf"] if fl else c["P"])[pos]
    return xr, xp


# ----------------------------------------------------------------------------- model
if HAVE_TORCH:
    class Net(nn.Module):
        """Paper head (GRU-128 over frames -> flatten -> BN -> Dense-1024 -> softmax) with optional pose branch."""

        def __init__(self, use_rgb, use_pose, dpose=68, K=K_FRAMES, h=128, nc=15, moddrop=0.0):
            super().__init__()
            self.use_rgb, self.use_pose, self.moddrop = use_rgb, use_pose, moddrop
            if use_rgb:
                self.rnorm = nn.LayerNorm(FEAT_DIM)
                self.rdrop = nn.Dropout(0.3)
                self.rgru = nn.GRU(FEAT_DIM, h, batch_first=True)
            if use_pose:
                self.pin = nn.Sequential(nn.Linear(dpose, 128), nn.ReLU(), nn.Dropout(0.2))
                self.pgru = nn.GRU(128, h, batch_first=True)
                self.register_buffer("pm", torch.zeros(dpose))
                self.register_buffer("ps", torch.ones(dpose))
            d = h * K * (int(use_rgb) + int(use_pose))
            self.head = nn.Sequential(nn.BatchNorm1d(d), nn.Linear(d, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, nc))

        def forward(self, xr=None, xp=None):
            zs = []
            if self.use_rgb:
                zs.append(self.rgru(self.rdrop(self.rnorm(xr)))[0])
            if self.use_pose:
                zs.append(self.pgru(self.pin((xp - self.pm) / self.ps))[0])
            if self.training and self.moddrop > 0 and len(zs) == 2:
                u = torch.rand(zs[0].shape[0], 1, 1, device=zs[0].device)
                zs[0] = zs[0] * (u >= self.moddrop).float()                       # RGB dropped when u < p
                zs[1] = zs[1] * ((u < self.moddrop) | (u >= 2 * self.moddrop)).float()  # pose dropped when p <= u < 2p
            return self.head(torch.cat(zs, 2).flatten(1))


def _to_dev(xr, xp, dev):
    r = torch.from_numpy(xr).to(dev).float() if xr is not None else None
    p = torch.from_numpy(xp).to(dev) if xp is not None else None
    return r, p


def predict_probs(model, data, ids, cfg, dev, tta, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(ids), bs):
            b = ids[s:s + bs]
            pr = 0
            for fl in ([False, True] if tta else [False]):
                xr, xp = make_batch(data, b, cfg, flip=fl)
                r, p = _to_dev(xr, xp, dev)
                pr = pr + torch.softmax(model(r, p), 1)
            out.append((pr / (2 if tta else 1)).cpu().numpy())
    return np.concatenate(out)


def train_run(data, split, C, cfg, seed, epochs=30, bs=64, lr=1e-3, dev="cuda", log=print):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    y = np.array([d["y"] for d in data])
    tr, va, te = [np.where(split == s)[0] for s in range(3)]
    model = Net(cfg.rgb, cfg.pose, data[0]["P"].shape[1], cfg.K, 128, C, cfg.moddrop).to(dev)
    if cfg.pose:  # z-score pose features with train statistics (orig + mirrored)
        allp = np.concatenate([data[i][k] for i in tr for k in ("P", "Pf")])
        model.pm.copy_(torch.from_numpy(allp.mean(0)))
        model.ps.copy_(torch.from_numpy(np.maximum(allp.std(0), 1e-3)))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * int(np.ceil(len(tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    best, best_state, best_ep = -1.0, None, -1
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(tr)
        tl = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            if len(b) < 2:
                continue
            xr, xp = make_batch(data, b, cfg, rng=rng, train=True)
            r, p = _to_dev(xr, xp, dev)
            yb = torch.from_numpy(y[b]).to(dev)
            loss = lossf(model(r, p), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tl += loss.item() * len(b)
        pv = predict_probs(model, data, va, cfg, dev, cfg.mirror)
        f1 = metrics(y[va], pv, C)["f1"]
        if f1 > best:
            best, best_ep = f1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"   {cfg.name} s{seed} ep{ep:02d} loss {tl / len(tr):.3f} val-F1 {f1:.4f} (best {best:.4f}@{best_ep}) {time.time() - t0:.0f}s")
    model.load_state_dict(best_state)
    return dict(val_probs=predict_probs(model, data, va, cfg, dev, cfg.mirror), val_y=y[va], val_ids=va,
                test_probs=predict_probs(model, data, te, cfg, dev, cfg.mirror), test_y=y[te], test_ids=te,
                best_ep=best_ep, n_train=len(tr))


def run_experiments(data, classes, out_dir, seeds=(0, 1, 2), cfgs=None, epochs=30, dev="cuda", log=print):
    os.makedirs(out_dir, exist_ok=True)
    cfgs = cfgs or CFGS
    C = len(classes)
    matches = [d["match"] for d in data]
    y = [d["y"] for d in data]
    for seed in seeds:
        split = make_split(matches, y, C, seed)
        cnt = np.stack([np.bincount(np.array(y)[split == s], minlength=C) for s in range(3)])
        log(f"split seed {seed}: clips train/val/test = {[(split == s).sum() for s in range(3)]}, "
            f"min per-class test count {cnt[2].min()}")
        for cfg in cfgs:
            fn = os.path.join(out_dir, f"{cfg.name}_s{seed}.npz")
            if os.path.exists(fn):
                log(f"skip {cfg.name} s{seed} (done)")
                continue
            r = train_run(data, split, C, cfg, seed, epochs, dev=dev, log=log)
            m = metrics(r["test_y"], r["test_probs"], C)
            np.savez(fn + ".tmp.npz", **r)
            os.replace(fn + ".tmp.npz", fn)
            log(f"== {cfg.name} seed {seed}: TEST acc {m['acc']:.4f} macro-F1 {m['f1']:.4f} (best epoch {r['best_ep']})")


# ----------------------------------------------------------------------------- reporting
def late_fusion(a, b):
    """Blend two runs' softmax outputs; weight chosen on validation macro-F1. a, b = loaded npz dicts."""
    C = a["val_probs"].shape[1]
    best, w = -1, 0.5
    for wt in np.linspace(0, 1, 11):
        f = metrics(a["val_y"], wt * a["val_probs"] + (1 - wt) * b["val_probs"], C)["f1"]
        if f > best:
            best, w = f, wt
    return w, w * a["test_probs"] + (1 - w) * b["test_probs"]


def report(out_dir, classes, seeds=(0, 1, 2), cfgs=None, pairs=(("Cover Drive", "Straight Drive"), ("Pull", "Hook")), show=True):
    import pandas as pd
    cfgs = cfgs or CFGS
    C = len(classes)
    res, probs = {}, {}
    for cfg in cfgs:
        for s in seeds:
            fn = os.path.join(out_dir, f"{cfg.name}_s{s}.npz")
            if os.path.exists(fn):
                z = dict(np.load(fn))
                probs[(cfg.name, s)] = z
                res[(cfg.name, s)] = metrics(z["test_y"], z["test_probs"], C)
    # late fusion rows
    for s in seeds:
        if ("rgb_m", s) in probs and ("pose_m", s) in probs:
            a, b = probs[("rgb_m", s)], probs[("pose_m", s)]
            w, pt = late_fusion(a, b)
            res[("late(rgb_m+pose_m)", s)] = metrics(a["test_y"], pt, C)
    names = list(dict.fromkeys(k[0] for k in res))
    rows = []
    for n in names:
        ss = [s for s in seeds if (n, s) in res]
        f1 = np.array([res[(n, s)]["f1"] for s in ss])
        acc = np.array([res[(n, s)]["acc"] for s in ss])
        d = [res[(n, s)]["f1"] - res[("rgb", s)]["f1"] for s in ss if ("rgb", s) in res]
        rows.append(dict(config=n, seeds=len(ss), acc=f"{100 * acc.mean():.2f} ± {100 * acc.std():.2f}",
                         macro_F1=f"{100 * f1.mean():.2f} ± {100 * f1.std():.2f}",
                         dF1_vs_rgb=f"{100 * np.mean(d):+.2f} ± {100 * np.std(d):.2f}" if d else "-"))
    table = pd.DataFrame(rows).set_index("config")
    pc = pd.DataFrame({n: np.mean([res[(n, s)]["per_class"] for s in seeds if (n, s) in res], 0) * 100 for n in names},
                      index=classes).round(1)
    pair_rows = []
    for A, B in pairs:
        if A in classes and B in classes:
            ia, ib = classes.index(A), classes.index(B)
            for n in names:
                e = []
                for s in seeds:
                    if (n, s) in res:
                        cm = np.array(res[(n, s)]["cm"])
                        e.append((cm[ia, ib] + cm[ib, ia]) / max(cm[ia].sum() + cm[ib].sum(), 1))
                pair_rows.append(dict(pair=f"{A} <-> {B}", config=n, confusion_rate_pct=round(100 * np.mean(e), 2)))
    pairs_df = pd.DataFrame(pair_rows)
    if show:
        print(table.to_string())
        print("\nper-class F1 (mean over splits):")
        print(pc.to_string())
        if len(pairs_df):
            print("\nmutual confusion rate:")
            print(pairs_df.pivot(index="config", columns="pair", values="confusion_rate_pct").to_string())
        else:
            print("\n(pair names not found; classes are:", classes, ")")
    return table, pc, pairs_df, res


def plot_confusions(res, classes, names, seed=0):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(names), figsize=(7 * len(names), 6.5))
    axs = np.atleast_1d(axs)
    for ax, n in zip(axs, names):
        cm = np.array(res[(n, seed)]["cm"], float)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=7)
        ax.set_yticklabels(classes, fontsize=7)
        ax.set_title(f"{n} (split {seed}, row-normalised)")
    plt.tight_layout()
    plt.show()
'''
open('/kaggle/working/cricfeat.py','w').write(fe); open('/kaggle/working/cricfuse.py','w').write(fu)
print(len(fe), len(fu), 'bytes written')

# ===== CELL 2: setup + SMOKE TEST (about 3-5 min). Send me the output. =====
import os, sys, glob, time, numpy as np, torch
sys.path.insert(0, '/kaggle/working')
import cricfeat as ce, cricfuse as cf
CACHE = '/kaggle/working/cache'          # Stage-1 output, same session
print(len(glob.glob(CACHE + '/chunk_*.pkl')), 'stage-1 chunks (expect 101)')
DEV = 'cuda'; print(torch.__version__, torch.cuda.get_device_name(0))
FEATS, RUNS = '/kaggle/working/feats', '/kaggle/working/runs'
t0 = time.time()
ce.extract(CACHE, '/kaggle/working/feats_smoke', DEV, limit_chunks=2)
print('feature extraction took', round(time.time() - t0), 's')
sdata, sclasses = cf.load_clips('/kaggle/working/feats_smoke')
t0 = time.time()
cf.run_experiments(sdata, sclasses, '/kaggle/working/runs_smoke', seeds=(0,), epochs=2, dev=DEV)
print('smoke training took', round(time.time() - t0), 's for', len(cf.CFGS), 'configs')
_ = cf.report('/kaggle/working/runs_smoke', sclasses, seeds=(0,))

# ===== CELL 3: FULL RUN (features -> 3 splits x 7 configs -> report). Start it and go to sleep. Resumable: re-run to continue. =====
t0 = time.time()
ce.extract(CACHE, FEATS, DEV)
print('features done in', round((time.time() - t0) / 60, 1), 'min', flush=True)
data, classes = cf.load_clips(FEATS)
t0 = time.time()
cf.run_experiments(data, classes, RUNS, seeds=(0, 1, 2), epochs=30, dev=DEV)
print('experiments took', round((time.time() - t0) / 60, 1), 'min')
table, pc, pairs, res = cf.report(RUNS, classes, seeds=(0, 1, 2))
table.to_csv('/kaggle/working/ablation_table.csv'); pc.to_csv('/kaggle/working/per_class_f1.csv'); pairs.to_csv('/kaggle/working/pair_confusions.csv')
cf.plot_confusions(res, classes, ['rgb', 'fuse_m_align'], seed=0)