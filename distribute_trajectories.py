#!/usr/bin/env python3
"""
distribute_trajectories.py

Разделяет DRAW-траекторию на N зон (по пространству), каждая зона назначается одному
дрону и остаётся за ним (минимизируем "перескакивание"). Балансируем нагрузку путем
перемещения граничных кусочков между соседними регионами.

Usage:
  python3 distribute_trajectories.py --input trajectory.txt --drones 4 --method spatial-regions --outdir trajectories
"""
import argparse
from pathlib import Path
import math
import numpy as np
from collections import namedtuple, defaultdict

# ------------------ ПАРАМЕТРЫ ------------------
CHUNK_MAX_LEN = 1.0        # макс длина кусочка (м)
DRAW_SPEED = 0.05          # скорость рисования (м/с)
TRANSIT_SPEED = 1.0        # скорость перелёта (м/с)
MIN_DIST = 0.20            # мин. горизонтальная дистанция (м) для конфликта
VERTICAL_MIN = 1.00        # мин. вертикальная дистанция (м) для конфликта
SIM_DT = 0.1               # шаг симуляции (с)
MAX_CONFLICT_RESOLVE_ITERS = 30

# Регион-балансирование
REGION_BALANCE_TOLERANCE = 0.12   # допустимое отклонение по времени от среднего (±12%)
REGION_BALANCE_MAX_ITERS = 200    # макс итераций перераспределения граничных чанков
# ------------------------------------------------

Chunk = namedtuple("Chunk", ["seg_idx", "chunk_idx", "points", "length", "centroid"])

# ----------------- Парсинг -----------------
def parse_trajectory_file(path):
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

# ----------------- Утилиты -----------------
def segment_length(seg):
    L = 0.0
    for i in range(1, len(seg)):
        L += math.hypot(seg[i][0] - seg[i-1][0], seg[i][1] - seg[i-1][1])
    return L

def segment_centroid(seg):
    xs = [p[0] for p in seg]; ys = [p[1] for p in seg]
    return (sum(xs)/len(xs), sum(ys)/len(ys))

# ----------------- Дробление -----------------
def split_segment_into_chunks(seg, max_len=CHUNK_MAX_LEN):
    if len(seg) < 2:
        return []
    chunks = []
    cur = [seg[0]]
    cur_len = 0.0
    for i in range(1, len(seg)):
        p0 = seg[i-1]; p1 = seg[i]
        edge_len = math.hypot(p1[0]-p0[0], p1[1]-p0[1])
        if edge_len == 0:
            continue
        remaining = edge_len
        dirx = (p1[0]-p0[0]) / edge_len
        diry = (p1[1]-p0[1]) / edge_len
        posx = p0[0]; posy = p0[1]
        z0 = p0[2]; z1 = p1[2]
        while remaining > 1e-9:
            space = max_len - cur_len
            if space <= 1e-9:
                chunks.append(cur)
                cur = [(posx, posy, z0)]
                cur_len = 0.0
                space = max_len
            take = min(space, remaining)
            posx += dirx * take
            posy += diry * take
            frac = (edge_len - remaining + take) / edge_len
            if z0 is not None and z1 is not None:
                posz = z0 + frac * (z1 - z0)
            else:
                posz = None
            cur.append((posx, posy, posz))
            cur_len += take
            remaining -= take
    if cur:
        cleaned = [cur[0]]
        for p in cur[1:]:
            if math.hypot(p[0]-cleaned[-1][0], p[1]-cleaned[-1][1]) > 1e-9:
                cleaned.append(p)
        if len(cleaned) >= 2:
            chunks.append(cleaned)
    out = []
    for ci,c in enumerate(chunks):
        L = segment_length(c)
        cx,cy = segment_centroid(c)
        out.append((ci, c, L, (cx,cy)))
    return out

def build_all_chunks(segments, max_len=CHUNK_MAX_LEN):
    chunks = []
    for si,seg in enumerate(segments):
        sub = split_segment_into_chunks(seg, max_len=max_len)
        for ci, pts, L, centroid in sub:
            chunks.append(Chunk(seg_idx=si, chunk_idx=ci, points=pts, length=L, centroid=centroid))
    return chunks

# ----------------- KMeans (numpy fallback) -----------------
def kmeans_numpy(points, k, max_iter=200):
    pts = np.array(points)
    if len(pts) == 0:
        return np.array([], dtype=int), np.zeros((0,2))
    k = min(k, len(pts))
    rng = np.random.default_rng(0)
    idx = rng.choice(len(pts), size=k, replace=False)
    centers = pts[idx].astype(float)
    for _ in range(max_iter):
        dists = np.sum((pts[:, None, :] - centers[None, :, :])**2, axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.array([pts[labels==i].mean(axis=0) if np.any(labels==i) else centers[i] for i in range(k)])
        if np.allclose(new_centers, centers):
            break
        centers = new_centers
    return labels, centers

# ----------------- Новая логика: spatial regions (1 region per drone) -----------------
def distribute_spatial_regions(chunks, k, draw_speed=DRAW_SPEED, tol=REGION_BALANCE_TOLERANCE, max_iters=REGION_BALANCE_MAX_ITERS):
    """
    1) Строим ровно k кластеров (kmeans).
    2) Вычисляем время в каждом регионе.
    3) Если сильный дисбаланс, перемещаем граничные куски из перегруженного региона
       в ближайший менее загруженный регион (итеративно), пока не достигнем tolerance.
    4) Возвращаем assignments: список списков глобальных индексов chunks (по drone).
    """
    n = len(chunks)
    if n == 0:
        return [[] for _ in range(k)]
    centroids = [c.centroid for c in chunks]
    # initial kmeans -> labels, centers
    try:
        from sklearn.cluster import KMeans
        X = np.array(centroids)
        km = KMeans(n_clusters=k, random_state=0).fit(X)
        labels = km.labels_.copy()
        centers = km.cluster_centers_.copy()
    except Exception:
        labels, centers = kmeans_numpy(centroids, k)
        centers = np.array([centroids[i] for i in range(len(centroids))]) if len(centroids) > 0 else np.zeros((k,2))
    # compute time per label
    def compute_times(labels):
        times = [0.0]*k
        for gi,lab in enumerate(labels):
            times[int(lab)] += chunks[gi].length / draw_speed
        return times
    times = compute_times(labels)
    total_time = sum(times)
    target = total_time / k if k>0 else 0.0
    it = 0
    # precompute neighbor candidate ordering by distance between chunk and cluster centers
    while it < max_iters:
        it += 1
        times = compute_times(labels)
        max_idx = int(np.argmax(times))
        min_idx = int(np.argmin(times))
        # stop if already balanced within tol
        if times[max_idx] <= target * (1.0 + tol):
            break
        # consider candidates from max_idx that are *closest* to min_idx center (i.e., boundary chunks)
        cand = [gi for gi in range(n) if int(labels[gi]) == max_idx]
        if not cand:
            # empty region — try to reinitialize one chunk into it
            # find largest region and move its smallest chunk to empty
            nonempty = [i for i in range(k) if i!=max_idx and any(int(labels[g])==i for g in range(n))]
            if not nonempty:
                break
            donor = int(np.argmax([sum(chunks[g].length for g in range(n) if int(labels[g])==i) for i in nonempty]))
            # pick smallest chunk in donor
            donor_chunks = [g for g in range(n) if int(labels[g])==donor]
            donor_chunks.sort(key=lambda g: chunks[g].length)
            if donor_chunks:
                labels[donor_chunks[0]] = max_idx
                continue
            else:
                break
        # sort candidates by distance to min cluster center (closest first)
        cent_min = centers[min_idx]
        cand_sorted = sorted(cand, key=lambda gi: (chunks[gi].centroid[0]-cent_min[0])**2 + (chunks[gi].centroid[1]-cent_min[1])**2)
        moved = False
        # move smallest pieces first among boundary candidates (less fragmentation)
        cand_sorted = sorted(cand_sorted, key=lambda gi: chunks[gi].length)
        for gi in cand_sorted:
            # move gi to nearest other cluster (prefer min_idx)
            # compute distances to all centers
            dists = [ (ci, (chunks[gi].centroid[0]-centers[ci][0])**2 + (chunks[gi].centroid[1]-centers[ci][1])**2) for ci in range(k) ]
            dists.sort(key=lambda x: x[1])
            # choose target cluster as the closest that is not current
            target_cluster = None
            for ci,_d in dists:
                if int(ci) != int(labels[gi]):
                    target_cluster = int(ci)
                    break
            if target_cluster is None:
                continue
            # apply tentative move and check improvement: moving should reduce max cluster time
            old_lab = int(labels[gi])
            labels[gi] = target_cluster
            new_times = compute_times(labels)
            # prefer moves that reduce the maximum region time
            if max(new_times) < max(times):
                moved = True
                # recompute centers to reflect change (simple recompute)
                # compute new centers as mean of centroids per label
                centers = np.zeros((k,2))
                counts = np.zeros(k, dtype=int)
                for idx,lab in enumerate(labels):
                    l = int(lab)
                    centers[l][0] += chunks[idx].centroid[0]
                    centers[l][1] += chunks[idx].centroid[1]
                    counts[l] += 1
                for ci in range(k):
                    if counts[ci] > 0:
                        centers[ci] /= counts[ci]
                    else:
                        # if center empty, set to a random chunk centroid to avoid NaN
                        centers[ci] = centroids[np.random.randint(len(centroids))]
                break
            else:
                # revert
                labels[gi] = old_lab
        if not moved:
            # если не удалось улучшить (нет подходящих граничных кусочков), выходим
            break
    # Final safety: ensure no empty clusters; if empty — assign smallest chunks from largest cluster
    counts = [0]*k
    for gi in range(n):
        counts[int(labels[gi])] += 1
    for ci in range(k):
        if counts[ci] == 0:
            # find donor cluster with largest total time
            times = compute_times(labels)
            donor = int(np.argmax(times))
            donor_chunks = [g for g in range(n) if int(labels[g])==donor]
            if not donor_chunks:
                continue
            donor_chunks.sort(key=lambda g: chunks[g].length)
            labels[donor_chunks[0]] = ci
            counts[ci] += 1
            counts[donor] -= 1
    # Build assignments: group chunk indices by label
    cluster_map = defaultdict(list)
    for gi, lab in enumerate(labels):
        cluster_map[int(lab)].append(gi)
    # To have stable mapping drone->region, sort clusters by centroid x then y and assign in that order
    final_centers = {}
    for lab, inds in cluster_map.items():
        if inds:
            xs = [chunks[i].centroid[0] for i in inds]; ys = [chunks[i].centroid[1] for i in inds]
            final_centers[lab] = (sum(xs)/len(xs), sum(ys)/len(ys))
        else:
            final_centers[lab] = (0.0, 0.0)
    # sort labels deterministically (left-to-right, then bottom-to-top)
    sorted_labels = sorted(list(cluster_map.keys()), key=lambda L: (final_centers[L][0], final_centers[L][1]))
    # produce assignments in drone order 0..k-1 mapped from sorted_labels
    assignments = [[] for _ in range(k)]
    for drone_idx, lab in enumerate(sorted_labels):
        assignments[drone_idx] = cluster_map[lab]
    # If there are less than k distinct labels (rare), pad empties
    for i in range(k):
        if i >= len(assignments):
            assignments.append([])
    return assignments

# ----------------- Merge chunks -> polylines -----------------
def merge_assigned_chunks_to_paths(chunks, assignments, sort_mode='spatial'):
    drone_paths = []
    for assigned in assignments:
        if not assigned:
            drone_paths.append([])
            continue
        if sort_mode == 'original':
            group = sorted(assigned, key=lambda gi: (chunks[gi].seg_idx, chunks[gi].chunk_idx))
        else:
            group = sorted(assigned, key=lambda gi: (chunks[gi].centroid[0], chunks[gi].centroid[1]))
        paths = []
        cur_path = []
        last_seg = None
        last_ci = None
        for gi in group:
            ch = chunks[gi]
            if last_seg is None:
                cur_path = [p for p in ch.points]
                last_seg = ch.seg_idx
                last_ci = ch.chunk_idx
            else:
                if ch.seg_idx == last_seg and ch.chunk_idx == last_ci + 1:
                    if math.hypot(cur_path[-1][0]-ch.points[0][0], cur_path[-1][1]-ch.points[0][1]) < 1e-9:
                        cur_path.extend(ch.points[1:])
                    else:
                        cur_path.extend(ch.points)
                    last_ci = ch.chunk_idx
                else:
                    if len(cur_path) >= 2:
                        paths.append(cur_path)
                    cur_path = [p for p in ch.points]
                    last_seg = ch.seg_idx
                    last_ci = ch.chunk_idx
        if cur_path and len(cur_path) >= 2:
            paths.append(cur_path)
        drone_paths.append(paths)
    return drone_paths

# ----------------- Timeline / conflicts (unchanged logic) -----------------
def build_timeline_for_drone(paths, z_draw_default, z_transit_default, transit_speed=TRANSIT_SPEED, draw_speed=DRAW_SPEED):
    events = []
    t = 0.0
    cur_pos = (0.0, 0.0, z_transit_default)
    for path in paths:
        if not path or len(path) < 2:
            continue
        sx, sy, sz = path[0]
        travel = math.hypot(sx - cur_pos[0], sy - cur_pos[1]) / transit_speed
        events.append((t, t+travel, "MOVE", cur_pos, (sx, sy, sz if sz is not None else z_draw_default), None))
        t += travel
        for i in range(1, len(path)):
            a = path[i-1]; b = path[i]
            az = a[2] if a[2] is not None else z_draw_default
            bz = b[2] if b[2] is not None else z_draw_default
            L = math.hypot(b[0]-a[0], b[1]-a[1])
            if L < 1e-9:
                continue
            dur = L / draw_speed
            events.append((t, t+dur, "DRAW", (a[0], a[1], az), (b[0], b[1], bz), None))
            t += dur
        cur_pos = (path[-1][0], path[-1][1], path[-1][2] if path[-1][2] is not None else z_draw_default)
    travel = math.hypot(cur_pos[0] - 0.0, cur_pos[1] - 0.0) / transit_speed
    events.append((t, t+travel, "MOVE", cur_pos, (0.0, 0.0, cur_pos[2]), None))
    t += travel
    return events, t

def sample_event_position(ev, t):
    t0, t1, mode, a, b, _ = ev
    if t1 <= t0:
        return a
    frac = (t - t0) / (t1 - t0)
    frac = max(0.0, min(1.0, frac))
    x = a[0] + frac * (b[0] - a[0])
    y = a[1] + frac * (b[1] - a[1])
    z = a[2] + frac * (b[2] - a[2])
    return (x,y,z)

def generate_time_samples(events, total_time, dt=SIM_DT):
    samples = []
    t = 0.0
    ev_idx = 0
    while t <= total_time + 1e-9:
        while ev_idx < len(events) and events[ev_idx][1] < t - 1e-9:
            ev_idx += 1
        if ev_idx >= len(events):
            last = events[-1]
            pos = last[4]
            samples.append((t, last[2], pos))
            t += dt
            continue
        ev = events[ev_idx]
        if ev[0] - 1e-9 <= t <= ev[1] + 1e-9:
            pos = sample_event_position(ev, t)
            samples.append((t, ev[2], pos))
        else:
            pos = ev[4]
            samples.append((t, "IDLE", pos))
        t += dt
    return samples

def detect_first_conflict(all_events_per_drone, all_total_times, dt=SIM_DT, min_dist=MIN_DIST, vertical_min=VERTICAL_MIN):
    T = max(all_total_times)
    samples = []
    for i, events in enumerate(all_events_per_drone):
        s = generate_time_samples(events, all_total_times[i], dt=dt)
        samples.append(s)
    steps = int(math.ceil(T / dt)) + 1
    for step in range(steps):
        t = step * dt
        positions = []
        modes = []
        for i in range(len(samples)):
            si = samples[i]
            idx = min(len(si)-1, int(round(t/dt)))
            tt, mode, pos = si[idx]
            positions.append(pos)
            modes.append(mode)
        n = len(positions)
        for a in range(n):
            for b in range(a+1, n):
                pa = positions[a]; pb = positions[b]
                horiz = math.hypot(pa[0]-pb[0], pa[1]-pb[1])
                vert = abs(pa[2]-pb[2])
                if modes[a] == "DRAW" and modes[b] == "DRAW":
                    if horiz < min_dist - 1e-9:
                        return (a,b,t,pa,pb,"horiz")
                    if vert < vertical_min - 1e-9:
                        return (a,b,t,pa,pb,"vert")
                if (modes[a] == "DRAW" and modes[b] == "MOVE") or (modes[a] == "MOVE" and modes[b] == "DRAW"):
                    if horiz < min_dist - 1e-9 and modes[a] == "DRAW" and modes[b] == "MOVE":
                        return (a,b,t,pa,pb,"horiz")
                    if horiz < min_dist - 1e-9 and modes[b] == "DRAW" and modes[a] == "MOVE":
                        return (b,a,t,pb,pa,"horiz")
    return None

def remaining_draw_length_after_time(events, t_query):
    rem = 0.0
    for ev in events:
        t0,t1,mode,a,b,_ = ev
        if t1 <= t_query + 1e-9:
            continue
        if mode == "DRAW":
            dur = max(0.0, t1 - max(t0, t_query))
            rem += dur * DRAW_SPEED
    return rem

def insert_move_at_time_for_drone(all_events_per_drone, drone_idx, insert_time, all_paths_per_drone, z_transit_default):
    events, total = build_timeline_for_drone(all_paths_per_drone[drone_idx], z_draw_default=1.0, z_transit_default=z_transit_default)
    ev_idx = None
    for ei, ev in enumerate(events):
        if ev[0] - 1e-9 <= insert_time <= ev[1] + 1e-9:
            ev_idx = ei
            break
    if ev_idx is None:
        return False
    ev = events[ev_idx]
    if ev[2] != "DRAW":
        pos = sample_event_position(ev, insert_time)
        all_paths_per_drone[drone_idx].insert(0, [(pos[0], pos[1], pos[2]), (pos[0], pos[1], pos[2])])
        return True
    pos = sample_event_position(ev, insert_time)
    paths = all_paths_per_drone[drone_idx]
    best = (None, None, 1e9, None)
    for pi, path in enumerate(paths):
        for vi in range(1, len(path)):
            a = path[vi-1]; b = path[vi]
            ax,ay = a[0], a[1]; bx,by = b[0], b[1]
            abx,aby = bx-ax, by-ay
            denom = abx*abx + aby*aby
            if denom < 1e-12:
                continue
            tproj = ((pos[0]-ax)*abx + (pos[1]-ay)*aby) / denom
            tproj_cl = max(0.0, min(1.0, tproj))
            projx = ax + tproj_cl * abx
            projy = ay + tproj_cl * aby
            d = math.hypot(pos[0]-projx, pos[1]-projy)
            if d < best[2]:
                best = (pi, vi, d, (projx, projy, a[2] if a[2] is not None else None))
    if best[0] is None:
        return False
    path_idx, vi, d, local_pos = best
    path = paths[path_idx]
    proj = local_pos
    left = path[:vi] + [proj]
    right = [proj] + path[vi:]
    new_paths = []
    for idx, p in enumerate(paths):
        if idx == path_idx:
            if len(left) >= 2:
                new_paths.append(left)
            new_paths.append([proj, proj])  # dummy MOVE
            if len(right) >= 2:
                new_paths.append(right)
        else:
            new_paths.append(p)
    all_paths_per_drone[drone_idx] = new_paths
    return True

def resolve_conflicts_iteratively(all_paths_per_drone, z_draw_default, z_transit_default):
    for attempt in range(MAX_CONFLICT_RESOLVE_ITERS):
        all_events = []
        totals = []
        for paths in all_paths_per_drone:
            evs, tot = build_timeline_for_drone(paths, z_draw_default=z_draw_default, z_transit_default=z_transit_default)
            all_events.append(evs)
            totals.append(tot)
        conflict = detect_first_conflict(all_events, totals, dt=SIM_DT, min_dist=MIN_DIST, vertical_min=VERTICAL_MIN)
        if conflict is None:
            return all_paths_per_drone, True
        a,b,t,pa,pb,reason = conflict
        rem_a = remaining_draw_length_after_time(all_events[a], t)
        rem_b = remaining_draw_length_after_time(all_events[b], t)
        if reason == "vert":
            to_move = a if pa[2] < pb[2] else b
        else:
            to_move = a if rem_a > rem_b else b
        ok = insert_move_at_time_for_drone(all_events, to_move, t, all_paths_per_drone, z_transit_default)
        if not ok:
            other = b if to_move == a else a
            ok2 = insert_move_at_time_for_drone(all_events, other, t, all_paths_per_drone, z_transit_default)
            if not ok2:
                print(f"Warning: couldn't insert MOVE for conflict between {a} and {b} at t={t:.3f}")
                return all_paths_per_drone, False
    print("Warning: reached max conflict resolution iterations")
    return all_paths_per_drone, False

# ----------------- Запись -----------------
def write_drone_files_from_paths(all_paths_per_drone, outdir, z_draw_default=1.0, z_transit_default=1.3):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for i, paths in enumerate(all_paths_per_drone):
        name = outdir / f"trajectory_drone_{i+1:02d}.txt"
        with open(name, 'w', encoding='utf-8') as f:
            f.write(f"MOVE 0.000 0.000 {z_transit_default:.3f}\n")
            for path in paths:
                if not path:
                    continue
                if len(path) == 2 and math.hypot(path[0][0]-path[1][0], path[0][1]-path[1][1]) < 1e-9:
                    x,y,z = path[0]
                    zt = z_transit_default if z is None else z
                    f.write(f"MOVE {x:.6f} {y:.6f} {zt:.3f}\n")
                    continue
                x0,y0,z0 = path[0]
                f.write(f"MOVE {x0:.6f} {y0:.6f} {z_transit_default:.3f}\n")
                for (x,y,z) in path:
                    zt = z_draw_default if z is None else z
                    f.write(f"DRAW {x:.6f} {y:.6f} {zt:.3f}\n")
                xl,yl,zl = path[-1]
                f.write(f"MOVE {xl:.6f} {yl:.6f} {z_transit_default:.3f}\n")
            f.write(f"MOVE 0.000 0.000 {z_transit_default:.3f}\n")
        print("Wrote", name)

# ----------------- main -----------------
def main():
    global CHUNK_MAX_LEN, DRAW_SPEED, TRANSIT_SPEED
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--drones', '-n', type=int, default=4)
    parser.add_argument('--method', choices=['round-robin','length-balance','kmeans','spatial-regions'], default='spatial-regions')
    parser.add_argument('--outdir', default='trajectories')
    parser.add_argument('--z_draw', type=float, default=1.0)
    parser.add_argument('--z_transit', type=float, default=1.3)
    parser.add_argument('--chunk', type=float, default=CHUNK_MAX_LEN)
    parser.add_argument('--draw_speed', type=float, default=DRAW_SPEED)
    parser.add_argument('--transit_speed', type=float, default=TRANSIT_SPEED)
    args = parser.parse_args()

    CHUNK_MAX_LEN = float(args.chunk)
    DRAW_SPEED = float(args.draw_speed)
    TRANSIT_SPEED = float(args.transit_speed)

    pts, segments = parse_trajectory_file(args.input)
    if not segments:
        print("No DRAW segments found in", args.input)
        return
    print("Found", len(segments), "draw segments")

    chunks = build_all_chunks(segments, max_len=CHUNK_MAX_LEN)
    if not chunks:
        print("No chunks produced")
        return
    print(f"Produced {len(chunks)} chunks (chunk max {CHUNK_MAX_LEN} m)")

    # назначение чанков на регионы/дронов
    if args.method == 'round-robin':
        # простая по чанкам
        assignments_idx = [[] for _ in range(args.drones)]
        for i in range(len(chunks)):
            assignments_idx[i % args.drones].append(i)
    elif args.method == 'kmeans':
        # kmeans по чанкам (k = drones)
        centroids = [c.centroid for c in chunks]
        try:
            from sklearn.cluster import KMeans
            X = np.array(centroids)
            km = KMeans(n_clusters=args.drones, random_state=0).fit(X)
            labels = km.labels_
        except Exception:
            labels, _ = kmeans_numpy(centroids, args.drones)
        assignments_idx = [[] for _ in range(args.drones)]
        for gi, lab in enumerate(labels):
            assignments_idx[int(lab)].append(gi)
    elif args.method == 'length-balance':
        # greedy по длине
        assignments_idx = [[] for _ in range(args.drones)]
        totals = [0.0]*args.drones
        order = sorted(range(len(chunks)), key=lambda i: -chunks[i].length)
        for gi in order:
            i = int(min(range(args.drones), key=lambda ii: totals[ii]))
            assignments_idx[i].append(gi)
            totals[i] += chunks[gi].length / DRAW_SPEED
    else:  # spatial-regions: ровно k регионов, перераспределение границ для баланса
        assignments_idx = distribute_spatial_regions(chunks, args.drones, draw_speed=DRAW_SPEED)

    # собрать polylines для каждого дрона (сортировка spatial -> локальность)
    paths_per_drone = merge_assigned_chunks_to_paths(chunks, assignments_idx, sort_mode='spatial')

    # попытаться разрешить конфликты (вставляем MOVE и т.д.)
    resolved_paths, ok = resolve_conflicts_iteratively(paths_per_drone, args.z_draw, args.z_transit)
    if not ok:
        print("Warning: conflict resolution incomplete")

    # записать файлы
    write_drone_files_from_paths(resolved_paths, args.outdir, z_draw_default=args.z_draw, z_transit_default=args.z_transit)
    print("Done.")

if __name__ == '__main__':
    main()
