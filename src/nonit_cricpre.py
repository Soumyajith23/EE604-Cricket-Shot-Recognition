%%writefile /kaggle/working/cricpre.py
"""
Stage 1 preprocessing for CricShot10k (pose-guided CricShotNet variant).

Per clip:
  1. Read all frames of the raw clip (~24 frames, 896x540).
  2. Multi-person pose (YOLO11-pose) + bat detector on every frame.
  3. Pick the striker with a Viterbi track over the whole clip (bat proximity + size/centrality prior
     + temporal consistency).  This replaces the unavailable 8-class player-type detector.
  4. Optional pose refinement on an aspect-preserving striker crop (better keypoints in wide shots).
  5. Crop (striker box U bat box, paper-style padding) -> 224x224 -> segmentation overlay (paper's seg model).
  6. Cache everything per chunk of clips: JPEG overlays, keypoints (raw-frame pixels), boxes, flags.

Clip order is a fixed random shuffle, so any prefix of finished chunks is a random (roughly stratified) subset.
Chunks are atomic and resumable.
"""
import os, re, sys, time, pickle, random, argparse
import numpy as np
import cv2

N_KP = 17
MAX_FRAMES = 64
JPEG_Q = 92
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


# ----------------------------------------------------------------------------- utilities
def _np(x):
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.asarray(x)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def list_clips(ds):
    """All clips under ds, deterministic order = fixed random shuffle (same in every shard)."""
    paths = []
    for dp, _, fs in os.walk(ds):
        for f in fs:
            if f.lower().endswith((".avi", ".mp4", ".mov")):
                paths.append(os.path.join(dp, f))
    paths.sort()
    rows = []
    for p in paths:
        m = re.search(r"vid(\d+)_(\d+)", os.path.basename(p))
        rows.append(dict(path=p, cls=os.path.basename(os.path.dirname(p)),
                         match=int(m.group(1)) if m else -1, clip=int(m.group(2)) if m else -1))
    random.Random(0).shuffle(rows)
    return rows


def read_frames(path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(path)
    fr = []
    while len(fr) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release()
    return fr


# ----------------------------------------------------------------------------- parsing model outputs
def parse_pose(res):
    if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
        return np.zeros((0, 4), np.float32), np.zeros((0, N_KP, 2), np.float32), np.zeros((0, N_KP), np.float32)
    b = _np(res.boxes.xyxy).astype(np.float32)
    kp = _np(res.keypoints.xy).astype(np.float32)
    kc = _np(res.keypoints.conf).astype(np.float32) if res.keypoints.conf is not None else np.ones(kp.shape[:2], np.float32)
    return b, kp, kc


def parse_bat(res):
    if res.boxes is None or len(res.boxes) == 0:
        return None
    b = _np(res.boxes.xyxy).astype(np.float32)
    c = _np(res.boxes.conf).astype(np.float32)
    return b[int(np.argmax(c))]


# ----------------------------------------------------------------------------- striker selection
def person_scores(boxes, kp, kc, bat, W, H):
    """Unary score per person: size/centrality prior + keypoint visibility + wrist-to-bat proximity."""
    k = len(boxes)
    if k == 0:
        return np.zeros(0, np.float32)
    h = np.maximum(boxes[:, 3] - boxes[:, 1], 1.0)
    w = np.maximum(boxes[:, 2] - boxes[:, 0], 1.0)
    area = (w * h) / float(W * H)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    prior = np.sqrt(area) * (1 - 0.5 * np.abs(cx - W / 2) / (W / 2))
    vis = (kc > 0.5).mean(1)
    s = 2.0 * prior + 0.5 * vis
    if bat is not None:
        bx1, by1, bx2, by2 = [float(v) for v in bat]
        barea = max((bx2 - bx1) * (by2 - by1), 1.0)
        for i in range(k):
            d = 1e9
            for j in (9, 10):  # wrists
                if kc[i, j] > 0.3:
                    x, y = kp[i, j]
                    d = min(d, float(np.hypot(max(bx1 - x, 0, x - bx2), max(by1 - y, 0, y - by2))))
            bs = max(0.0, 1.0 - (d / h[i]) / 0.35) if d < 1e8 else 0.0
            ix = max(0.0, min(bx2, boxes[i, 2]) - max(bx1, boxes[i, 0]))
            iy = max(0.0, min(by2, boxes[i, 3]) - max(by1, boxes[i, 1]))
            bs = max(bs, 0.5 * (ix * iy) / barea)  # bat inside the person's box even if wrists are occluded
            s[i] += 3.0 * bs
    return s


def _trans_cost(a, b):
    ha, hb = a[3] - a[1], b[3] - b[1]
    d = np.hypot((a[0] + a[2]) / 2 - (b[0] + b[2]) / 2, (a[1] + a[3]) / 2 - (b[1] + b[3]) / 2)
    d = d / max(0.5 * (ha + hb), 1.0)
    return 1.5 * (1 - iou(a, b)) + min(1.0, float(d))


def track_striker(boxes_l, scores_l, person_bias=0.4, p2n=0.6, n2n=0.1):
    """Viterbi over frames. State per frame = one of its persons, or 'none'. Returns index per frame (-1 = none)."""
    T = len(boxes_l)
    best, back = [], []
    for t in range(T):
        k = len(boxes_l[t])
        sc = np.concatenate([np.asarray(scores_l[t], np.float64) - person_bias, [0.0]])
        if t == 0:
            best.append(sc)
            back.append(np.full(k + 1, -1))
            continue
        kp_ = len(boxes_l[t - 1])
        prev = best[-1]
        cur = np.empty(k + 1)
        bk = np.empty(k + 1, dtype=int)
        for s in range(k + 1):
            cand = np.empty(kp_ + 1)
            for s2 in range(kp_ + 1):
                if s < k and s2 < kp_:
                    c = _trans_cost(boxes_l[t - 1][s2], boxes_l[t][s])
                elif s == k and s2 == kp_:
                    c = n2n
                else:
                    c = p2n
                cand[s2] = prev[s2] - c
            bk[s] = int(np.argmax(cand))
            cur[s] = cand[bk[s]] + sc[s]
        best.append(cur)
        back.append(bk)
    s = int(np.argmax(best[-1]))
    out = [-1] * T
    for t in range(T - 1, -1, -1):
        k = len(boxes_l[t])
        out[t] = s if s < k else -1
        s = int(back[t][s])
    return out


# ----------------------------------------------------------------------------- cropping
def make_crop_box(sbox, bat, W, H):
    """Paper-style crop: striker box U bat box (only if the bat is plausibly in the striker's hands), +10% x pad, +5% y pad."""
    x1, y1, x2, y2 = [float(v) for v in sbox]
    if bat is not None:
        w, h = x2 - x1, y2 - y1
        bcx, bcy = (bat[0] + bat[2]) / 2, (bat[1] + bat[3]) / 2
        if x1 - 0.4 * w <= bcx <= x2 + 0.4 * w and y1 - 0.4 * h <= bcy <= y2 + 0.4 * h:
            x1, y1, x2, y2 = min(x1, bat[0]), min(y1, bat[1]), max(x2, bat[2]), max(y2, bat[3])
    w, h = x2 - x1, y2 - y1
    x1 -= 0.10 * w
    x2 += 0.10 * w
    y1 -= 0.05 * h
    y2 += 0.05 * h
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(W, int(np.ceil(x2))), min(H, int(np.ceil(y2)))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return (x1, y1, x2, y2)


def _jpg(im):
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])[1].tobytes()


# ----------------------------------------------------------------------------- one clip
def process_clip(frames, M, refine=True, save_raw=False):
    N = len(frames)
    H, W = frames[0].shape[:2]
    pres = M["pose"](frames, imgsz=640, half=True, verbose=False)
    bres = M["bat"](frames, max_det=1, half=True, verbose=False)
    dets = [parse_pose(r) for r in pres]
    bats = [parse_bat(r) for r in bres]

    scores_l = [person_scores(d[0], d[1], d[2], bats[t], W, H) for t, d in enumerate(dets)]
    sel = track_striker([d[0] for d in dets], scores_l)
    valid = np.array([s >= 0 for s in sel])
    if not valid.any():
        return dict(ok=False, reason="no_striker", n=N, W=W, H=H)

    sbox = np.zeros((N, 4), np.float32)
    kp = np.zeros((N, N_KP, 2), np.float32)
    kc = np.zeros((N, N_KP), np.float32)
    for t, s in enumerate(sel):
        if s >= 0:
            sbox[t], kp[t], kc[t] = dets[t][0][s], dets[t][1][s], dets[t][2][s]
    vi = np.where(valid)[0]
    for t in range(N):  # frames without the striker keep the nearest valid box (flagged by `valid`)
        if not valid[t]:
            sbox[t] = sbox[vi[np.argmin(np.abs(vi - t))]]

    n_refined = 0
    if refine:  # re-run pose on an aspect-preserving, upscaled crop around the striker
        crops, meta = [], []
        for t in vi:
            b = sbox[t]
            w, h = b[2] - b[0], b[3] - b[1]
            x1, y1 = int(max(0, b[0] - 0.2 * w)), int(max(0, b[1] - 0.2 * h))
            x2, y2 = int(min(W, b[2] + 0.2 * w)), int(min(H, b[3] + 0.2 * h))
            if x2 - x1 >= 8 and y2 - y1 >= 8:
                crops.append(frames[t][y1:y2, x1:x2])
                meta.append((t, x1, y1))
        if crops:
            rres = M["pose"](crops, imgsz=384, half=True, verbose=False)
            for (t, x1, y1), r in zip(meta, rres):
                bb, kk, cc = parse_pose(r)
                if len(bb) == 0:
                    continue
                ob = sbox[t] - np.array([x1, y1, x1, y1], np.float32)
                ious = [iou(b_, ob) for b_ in bb]
                j = int(np.argmax(ious))
                if ious[j] > 0.3:
                    kp[t] = kk[j] + np.array([x1, y1], np.float32)
                    kc[t] = cc[j]
                    n_refined += 1

    cbox = np.zeros((N, 4), np.int32)
    bat_arr = np.full((N, 4), -1, np.float32)
    crops224 = []
    for t in range(N):
        if bats[t] is not None:
            bat_arr[t] = bats[t]
        cb = make_crop_box(sbox[t], bats[t], W, H) or (0, 0, W, H)
        cbox[t] = cb
        crops224.append(cv2.resize(frames[t][cb[1]:cb[3], cb[0]:cb[2]], (224, 224)))

    sres = M["seg"](crops224, imgsz=256, max_det=2, half=True, verbose=False)
    seg = [np.ascontiguousarray(r.plot(boxes=False, probs=False, labels=False)) for r in sres]
    nmask = np.array([0 if r.masks is None else len(r.masks) for r in sres], np.int8)

    rec = dict(ok=True, n=N, W=W, H=H, jpg_seg=[_jpg(im) for im in seg], kp=kp, kc=kc, valid=valid,
               sbox=sbox, cbox=cbox, bat=bat_arr, nmask=nmask, n_refined=n_refined)
    if save_raw:
        rec["jpg_raw"] = [_jpg(im) for im in crops224]
    return rec


# ----------------------------------------------------------------------------- shard runner
def run_shard(rows, M, out, shard=0, nshards=1, chunk=100, refine=True, save_raw=False, limit=0,
              log=print, max_consec_fail=15):
    os.makedirs(out, exist_ok=True)
    if limit:
        rows = rows[:limit]
    nchunks = (len(rows) + chunk - 1) // chunk
    mine = [c for c in range(nchunks) if c % nshards == shard]
    t0, done, consec = time.time(), 0, 0
    for c in mine:
        fn = os.path.join(out, f"chunk_{c:04d}.pkl")
        if os.path.exists(fn):
            log(f"skip existing {os.path.basename(fn)}")
            continue
        recs = []
        for r in rows[c * chunk:(c + 1) * chunk]:
            try:
                fr = read_frames(r["path"])
                rec = dict(ok=False, reason="short", n=len(fr)) if len(fr) < 5 else process_clip(fr, M, refine, save_raw)
            except Exception as e:  # keep going; a systematic bug is caught below
                rec = dict(ok=False, reason="exc:" + repr(e)[:200])
            consec = consec + 1 if str(rec.get("reason", "")).startswith("exc:") else 0
            if consec >= max_consec_fail:
                raise RuntimeError(f"{consec} consecutive clip failures, last: {rec['reason']}")
            rec.update(r)
            recs.append(rec)
            done += 1
        tmp = fn + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(recs, f, protocol=4)
        os.replace(tmp, fn)
        el = time.time() - t0
        log(f"shard {shard} chunk {c} done | {done} clips | {el / done:.2f}s/clip | ok {sum(bool(x['ok']) for x in recs)}/{len(recs)}")
    return done


# ----------------------------------------------------------------------------- QA montage
def montage(rec, n_show=8, size=224):
    """Grid image: top row = segmentation overlay, bottom row = overlay + keypoints (crop coordinates)."""
    n = rec["n"]
    idx = np.linspace(0, n - 1, min(n_show, n)).round().astype(int)
    top, bot = [], []
    for t in idx:
        im = cv2.imdecode(np.frombuffer(rec["jpg_seg"][t], np.uint8), cv2.IMREAD_COLOR)
        top.append(im.copy())
        x1, y1, x2, y2 = rec["cbox"][t]
        sx, sy = size / max(x2 - x1, 1), size / max(y2 - y1, 1)
        pts = [((x - x1) * sx, (y - y1) * sy, c) for (x, y), c in zip(rec["kp"][t], rec["kc"][t])]
        for a, b in SKELETON:
            if pts[a][2] > 0.3 and pts[b][2] > 0.3:
                cv2.line(im, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), (0, 255, 255), 2)
        for x, y, c in pts:
            if c > 0.3:
                cv2.circle(im, (int(x), int(y)), 3, (0, 0, 255), -1)
        cv2.putText(im, f"{t}{'' if rec['valid'][t] else ' (no striker)'}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        bot.append(im)
    return np.vstack([np.hstack(top), np.hstack(bot)])


# ----------------------------------------------------------------------------- cache summary
def summarize(out):
    """Per-class QA of a cache directory, read chunk by chunk (never holds the whole cache in RAM)."""
    import glob, collections
    per = collections.defaultdict(lambda: [0, 0, 0.0, 0.0])  # n, ok, sum valid-frac, sum bat-frac
    fails = collections.Counter()
    size, nfiles = 0, 0
    for f in sorted(glob.glob(os.path.join(out, "chunk_*.pkl"))):
        size += os.path.getsize(f)
        nfiles += 1
        for r in pickle.load(open(f, "rb")):
            p = per[r["cls"]]
            p[0] += 1
            if r["ok"]:
                p[1] += 1
                p[2] += float(r["valid"].mean())
                p[3] += float((r["bat"][:, 0] >= 0).mean())
            else:
                fails[str(r.get("reason"))[:60]] += 1
    tot = sum(p[0] for p in per.values())
    ok = sum(p[1] for p in per.values())
    print(f"{nfiles} chunks | {tot} clips cached | ok {ok} ({100 * ok / max(tot, 1):.1f}%) | {size / 1e9:.2f} GB")
    print(f"{'class':18s} {'n':>5s} {'ok%':>6s} {'striker-frames%':>16s} {'bat-frames%':>12s}")
    for c in sorted(per):
        n, k, v, b = per[c]
        print(f"{c:18s} {n:5d} {100 * k / max(n, 1):6.1f} {100 * v / max(k, 1):16.1f} {100 * b / max(k, 1):12.1f}")
    if fails:
        print("failure reasons:", dict(fails))
    return per


# ----------------------------------------------------------------------------- main
def load_models(pose, bat, seg):
    from ultralytics import YOLO
    M = dict(pose=YOLO(pose), bat=YOLO(bat), seg=YOLO(seg))
    dummy = np.zeros((540, 896, 3), np.uint8)
    M["pose"]([dummy], imgsz=640, half=True, verbose=False)
    M["bat"]([dummy], max_det=1, half=True, verbose=False)
    M["seg"]([np.zeros((224, 224, 3), np.uint8)], imgsz=256, max_det=2, half=True, verbose=False)
    return M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pose", required=True)
    ap.add_argument("--bat", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--refine", type=int, default=1)
    ap.add_argument("--save_raw", type=int, default=0)
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)  # before torch is imported
    log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
    rows = list_clips(a.ds)
    log(f"{len(rows)} clips | shard {a.shard}/{a.nshards} | gpu {a.gpu}")
    M = load_models(a.pose, a.bat, a.seg)
    n = run_shard(rows, M, a.out, a.shard, a.nshards, a.chunk, bool(a.refine), bool(a.save_raw), a.limit, log)
    log(f"finished, {n} clips processed in this run")


if __name__ == "__main__":
    main()