#!/usr/bin/env python3
"""
schedule_resolver.py

Читает per-drone files in indir (trajectory_drone_XX.txt),
симулирует движение (speed м/с), и вычисляет стартовые задержки (schedules.json),
чтобы во времени не возникало конфликтов по расстоянию < safety.

Outputs updated schedules.json in the same directory.
"""
import argparse
from pathlib import Path
import math
import json

def parse_drone_file(path):
    pts = []
    with open(path,'r',encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s: continue
            parts = s.split()
            mode = parts[0].upper()
            if len(parts) >= 3 and mode in ("MOVE","DRAW"):
                x = float(parts[1]); y = float(parts[2])
                pts.append((mode,x,y))
    # convert into motion primitives: sequence of (type, list_of_points)
    segs = []
    cur = None
    for m,x,y in pts:
        if m=="MOVE":
            # MOVE marks transit; if currently building draw, close it
            if cur and cur[0]=="DRAW":
                segs.append(cur); cur=None
            # start/append a MOVE primitive
            segs.append(("MOVE", [(x,y)]))
        else: # DRAW
            if cur and cur[0]=="DRAW":
                cur[1].append((x,y))
            else:
                cur = ("DRAW", [(x,y)])
    if cur:
        segs.append(cur)
    # flatten consecutive MOVE points to path
    path = []
    for item in segs:
        if item[0]=="MOVE":
            # path append as transit step from last point to this point
            path.append(("MOVE", item[1][0]))
        else:
            # DRAW: sequence of points
            for pt in item[1]:
                path.append(("DRAW", pt))
    return path

def build_time_trajectory(path, speed, start_delay=0.0):
    # path: list of ("MOVE"|"DRAW", (x,y))
    # We'll simulate: for MOVE between consecutive points, travel at speed; for DRAW between consecutive DRAW pts, travel at speed too.
    # Output: list of (t, x, y) sampled at each event point (not continuous). We'll later interpolate.
    events = []
    t = start_delay
    last = (0.0,0.0)  # assume launch point at origin
    first = True
    for item in path:
        typ, (x,y) = item
        if first and typ=="MOVE":
            # move from origin to first move
            dist = math.hypot(x-last[0], y-last[1])
            dt = dist / speed if speed>0 else 0.0
            t += dt
            events.append((t,x,y))
            last = (x,y)
            first=False
            continue
        # for subsequent items: move from last to this
        dist = math.hypot(x-last[0], y-last[1])
        dt = dist / speed if speed>0 else 0.0
        t += dt
        events.append((t,x,y))
        last = (x,y)
        first=False
    return events

def sample_positions(events, dt, total_time):
    # events: list of (t,x,y) sorted
    # sample at times 0..total_time step dt, linearly interpolate between events
    samples = []
    if not events:
        return samples
    times = [e[0] for e in events]
    xs = [e[1] for e in events]; ys=[e[2] for e in events]
    t=0.0
    i=0
    N = len(events)
    while t <= total_time + 1e-6:
        # find segment covering t (events[i-1], events[i])
        while i < N and times[i] < t: i+=1
        if i==0:
            x = xs[0]; y=ys[0]
        elif i>=N:
            x = xs[-1]; y=ys[-1]
        else:
            t0 = times[i-1]; t1 = times[i]
            if t1==t0:
                x=xs[i]; y=ys[i]
            else:
                alpha = (t - t0) / (t1 - t0)
                x = xs[i-1] + alpha*(xs[i]-xs[i-1])
                y = ys[i-1] + alpha*(ys[i]-ys[i-1])
        samples.append((t,x,y))
        t += dt
    return samples

def detect_conflicts(samples_a, samples_b, safety):
    # assume same sampling times
    for (ta, xa, ya), (tb, xb, yb) in zip(samples_a, samples_b):
        if abs(ta-tb) > 1e-6:
            # times misaligned; skip or handle
            pass
        d = math.hypot(xa-xb, ya-yb)
        if d < safety:
            return True
    return False

def simulate_and_resolve(indir, speed=0.5, safety=0.15, dt=0.1, max_iter=200):
    indir = Path(indir)
    files = sorted(indir.glob("trajectory_drone_*.txt"))
    N = len(files)
    paths = [parse_drone_file(f) for f in files]
    # initial delays from schedules.json if exists
    sched_path = indir/"schedules.json"
    if sched_path.exists():
        sched = json.loads(sched_path.read_text())
    else:
        sched = {f.stem: 0.0 for f in files}

    delays = [0.0]*N
    # map filenames order to keys in sched (best effort)
    for i,f in enumerate(files):
        key = f.stem
        if key in sched:
            delays[i] = float(sched[key])
        else:
            # try drone_01 etc
            delays[i] = 0.0

    # construct event timelines
    events_list = [build_time_trajectory(p, speed, start_delay=delays[i]) for i,p in enumerate(paths)]
    total_times = [ev[-1][0] if ev else 0.0 for ev in events_list]
    tot = max(total_times) + 1.0
    # iterative conflict resolution: sample at dt and push delays if needed
    iter_count = 0
    while iter_count < max_iter:
        iter_count += 1
        # sample every drone positions
        samples = [sample_positions(events_list[i], dt, max(total_times)) for i in range(N)]
        conflict_found = False
        for i in range(N):
            for j in range(i+1, N):
                # align sample lengths
                L = min(len(samples[i]), len(samples[j]))
                if L==0: continue
                s_i = samples[i][:L]; s_j = samples[j][:L]
                if detect_conflicts(s_i, s_j, safety):
                    conflict_found = True
                    # decide which drone to delay: delay the one with larger total remaining time (so we don't delay the quicker one)
                    rem_i = total_times[i] - samples[i][-1][0] if samples[i] else total_times[i]
                    rem_j = total_times[j] - samples[j][-1][0] if samples[j] else total_times[j]
                    # simpler: delay the one with larger remaining 'work' (or larger index)
                    if rem_i >= rem_j:
                        # delay drone i by delta = safety / speed (approx) plus small margin
                        delta = max(0.5, safety / max(1e-6, speed))
                        delays[i] += delta
                        # rebuild its events
                        events_list[i] = build_time_trajectory(paths[i], speed, start_delay=delays[i])
                        total_times[i] = events_list[i][-1][0] if events_list[i] else 0.0
                    else:
                        delta = max(0.5, safety / max(1e-6, speed))
                        delays[j] += delta
                        events_list[j] = build_time_trajectory(paths[j], speed, start_delay=delays[j])
                        total_times[j] = events_list[j][-1][0] if events_list[j] else 0.0
                    # break pair loops to restart checking
                    break
            if conflict_found:
                break
        if not conflict_found:
            print("No conflicts found after", iter_count, "iterations.")
            break
    if conflict_found:
        print("Stopped after max_iter; conflicts may remain.")
    # write updated schedules.json (map file stem -> delay)
    sched_out = {}
    for i,f in enumerate(files):
        sched_out[f.stem] = delays[i]
    with open(indir/"schedules.json",'w',encoding='utf-8') as f:
        json.dump(sched_out, f, indent=2)
    print("Wrote updated schedules.json")
    # print summary
    for i,f in enumerate(files):
        print(f"{f.name}: delay={delays[i]:.1f}s final_end={events_list[i][-1][0] if events_list[i] else 0.0:.1f}s")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', required=True)
    parser.add_argument('--speed', type=float, default=0.5)
    parser.add_argument('--safety', type=float, default=0.15)
    parser.add_argument('--dt', type=float, default=0.1)
    parser.add_argument('--max_iter', type=int, default=200)
    args = parser.parse_args()
    simulate_and_resolve(args.indir, speed=args.speed, safety=args.safety, dt=args.dt, max_iter=args.max_iter)
