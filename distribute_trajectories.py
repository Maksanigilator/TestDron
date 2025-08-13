#!/usr/bin/env python3
"""
distribute_trajectories.py

Распределяет сегменты из одного trajectory.txt на N дронов и сохраняет
trajectory_drone_01.txt ... trajectory_drone_NN.txt

Usage:
  python distribute_trajectories.py --input trajectory.txt --drones 4 --method kmeans --outdir trajectories

Methods:
  - round-robin
  - length-balance  (greedy assign to drone with minimal total length)
  - kmeans          (cluster segments by centroid; requires scikit-learn, fallback to numpy kmeans)

Outputs:
  outdir/trajectory_drone_01.txt ...
"""
import argparse
from pathlib import Path
import math
import numpy as np

def parse_trajectory_file(path):
    """
    Возвращает список точек с mode,x,y,(z or None) и также список draw-сегментов.
    draw_segments = [ [ (x,y,z_or_None), ... ], ... ]
    """
    pts = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            mode = parts[0].upper()
            if mode in ("MOVE","DRAW"):
                if len(parts) >= 3:
                    x = float(parts[1]); y = float(parts[2])
                    z = None
                    if len(parts) >= 4:
                        try:
                            z = float(parts[3])
                        except:
                            z = None
                    pts.append({"mode": mode, "x": x, "y": y, "z": z})
    # extract draw segments: contiguous runs of DRAW points
    segments = []
    cur = []
    for p in pts:
        if p["mode"] == "DRAW":
            cur.append((p["x"], p["y"], p["z"]))
        else:
            if cur:
                segments.append(cur)
                cur = []
    if cur:
        segments.append(cur)
    return pts, segments

def segment_length(seg):
    L = 0.0
    for i in range(1, len(seg)):
        dx = seg[i][0]-seg[i-1][0]; dy = seg[i][1]-seg[i-1][1]
        L += math.hypot(dx,dy)
    return L

def segment_centroid(seg):
    xs = [p[0] for p in seg]; ys = [p[1] for p in seg]
    return (sum(xs)/len(xs), sum(ys)/len(ys))

def distribute_round_robin(segments, k):
    out = [[] for _ in range(k)]
    for i,seg in enumerate(segments):
        out[i % k].append(seg)
    return out

def distribute_length_balance(segments, k):
    totals = [0.0]*k
    out = [[] for _ in range(k)]
    for seg in sorted(segments, key=lambda s: -segment_length(s)):  # assign long ones first
        L = segment_length(seg)
        i = int(min(range(k), key=lambda ii: totals[ii]))
        out[i].append(seg)
        totals[i] += L
    return out

def kmeans_numpy(points, k, max_iter=100):
    # simple kmeans on pts Nx2
    pts = np.array(points)
    # init: random choose k distinct
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=k, replace=False)
    centers = pts[idx].astype(float)
    for it in range(max_iter):
        # assign
        dists = np.sum((pts[:,None,:]-centers[None,:,:])**2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([pts[labels==i].mean(axis=0) if np.any(labels==i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers

def distribute_kmeans(segments, k):
    centroids = [segment_centroid(s) for s in segments]
    # try sklearn
    try:
        from sklearn.cluster import KMeans
        X = np.array(centroids)
        km = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = km.labels_
    except Exception:
        labels, _ = kmeans_numpy(centroids, k)
    out = [[] for _ in range(k)]
    for seg,lab in zip(segments, labels):
        out[int(lab)].append(seg)
    return out

def write_drone_files(assignments, outdir, z_draw_default=None, z_transit_default=1.3):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, segs in enumerate(assignments):
        name = outdir / f"trajectory_drone_{i+1:02d}.txt"
        with open(name, 'w', encoding='utf-8') as f:
            f.write(f"MOVE 0.000 0.000 {z_transit_default:.3f}\n")
            for seg in segs:
                x0,y0,z0 = seg[0]
                zb = z0 if z0 is not None else z_draw_default
                # transit to first point (keep z_transit line, then z_draw)
                f.write(f"MOVE {x0:.6f} {y0:.6f}\n")
                f.write(f"MOVE {x0:.6f} {y0:.6f}\n")  # second MOVE here kept for compatibility (visualizer ignores z)
                # DRAW lines
                for (x,y,z) in seg:
                    f.write(f"DRAW {x:.6f} {y:.6f}\n")
                # after segment, transit away (no z here)
                xl,yl,zl = seg[-1]
                f.write(f"MOVE {xl:.6f} {yl:.6f}\n")
            f.write(f"MOVE 0.000 0.000 {z_transit_default:.3f}\n")
        print("Wrote", name)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--drones', '-n', type=int, default=4)
    parser.add_argument('--method', choices=['round-robin','length-balance','kmeans'], default='length-balance')
    parser.add_argument('--outdir', default='trajectories')
    parser.add_argument('--z_draw', type=float, default=1.0)
    args = parser.parse_args()

    pts, segments = parse_trajectory_file(args.input)
    if not segments:
        print("No DRAW segments found in", args.input)
        return
    print("Found", len(segments), "draw segments")

    if args.method == 'round-robin':
        assignments = distribute_round_robin(segments, args.drones)
    elif args.method == 'kmeans':
        assignments = distribute_kmeans(segments, args.drones)
    else:
        assignments = distribute_length_balance(segments, args.drones)

    write_drone_files(assignments, args.outdir, z_draw_default=args.z_draw)

if __name__ == '__main__':
    main()
