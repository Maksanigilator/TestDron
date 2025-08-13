#!/usr/bin/env python3
"""
trajectory_to_commands.py

Reads trajectory.txt (lines: "MOVE x y" or "DRAW x y") and config.json,
decomposes movements so that set_position calls have max step <= max_speed*dt
(assumes set_position called at 1/dt Hz), and writes a plain commands file:

import rospy
r = rospy.Rate(10)
set_position(x=..., y=..., z=...)
r.sleep()

All servo calls removed as requested.
"""
import argparse
import json
import math
from pathlib import Path
import re

def load_config(path: Path):
    txt = path.read_text(encoding='utf-8')
    # allow // and /* */ comments removal
    txt = re.sub(r'//.*', '', txt)
    txt = re.sub(r'/\*.*?\*/', '', txt, flags=re.S)
    return json.loads(txt)

def parse_traj(path: Path):
    traj = []
    with path.open('r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) >= 3 and parts[0].upper() in ("MOVE","DRAW"):
                try:
                    mode = parts[0].upper()
                    x = float(parts[1]); y = float(parts[2])
                    traj.append((mode, x, y))
                except:
                    continue
            else:
                # try to extract x=.., y=..
                m = re.search(r"x\s*=\s*([-+]?\d*\.?\d+)", s)
                n = re.search(r"y\s*=\s*([-+]?\d*\.?\d+)", s)
                if m and n:
                    mode = "DRAW" if "draw" in s.lower() or "#draw" in s.lower() else "MOVE"
                    traj.append((mode, float(m.group(1)), float(n.group(1))))
    return traj

def decompose_segment(cur_x, cur_z, tx, tz, max_step):
    """
    Decompose movement from (cur_x,cur_z) to (tx,tz) into N steps so that each step length <= max_step.
    Returns list of (x,z) intermediate positions including final target (but NOT including starting point).
    """
    dx = tx - cur_x
    dz = tz - cur_z
    dist = math.hypot(dx, dz)
    if dist == 0.0:
        return [(tx, tz)]
    steps = int(math.ceil(dist / max_step))
    if steps < 1:
        steps = 1
    pts = []
    for s in range(1, steps + 1):
        t = s / steps
        xi = cur_x + dx * t
        zi = cur_z + dz * t
        pts.append((xi, zi))
    return pts

def format_f(x, prec):
    return f"{x:.{prec}f}"

def generate_commands(traj, cfg):
    offset_x = float(cfg.get("offset_x", 0.0))
    offset_z = float(cfg.get("offset_z", 0.0))
    default_y = float(cfg.get("default_y", 0.0))
    dt = float(cfg.get("dt", 0.1))
    max_speed = float(cfg.get("max_speed", 0.5))
    hold_draw = int(cfg.get("hold_draw", 2))
    hold_move = int(cfg.get("hold_move", 1))
    start_from_origin = bool(cfg.get("start_from_origin", False))
    prec = int(cfg.get("float_precision", 6))

    max_step = max_speed * dt  # max distance between set_position calls

    out_lines = []
    # add rospy rate header
    out_lines.append("import rospy")
    out_lines.append("r = rospy.Rate(10)")
    out_lines.append("")

    if not traj:
        return out_lines

    # compute transformed targets list (mode, x_out, y_out, z_out)
    targets = []
    for mode, xin, yin in traj:
        xout = xin + offset_x
        zout = yin + offset_z  # y_in -> z_out
        yout = default_y
        targets.append((mode, xout, yout, zout))

    # current position: if start_from_origin True -> origin (0,default_y,0)
    # else assume current position equals first target (so no extra movement before first)
    if start_from_origin:
        cur_x = 0.0
        cur_z = 0.0
        cur_y = default_y
    else:
        cur_x = targets[0][1]
        cur_z = targets[0][3]
        cur_y = targets[0][2]

    prev_mode = None

    for idx, (mode, tx, ty, tz) in enumerate(targets):
        # we no longer output servo toggles; keep mode for hold logic only
        if prev_mode is None:
            prev_mode = mode
        else:
            prev_mode = mode

        # decompose movement from (cur_x,cur_z) to (tx,tz)
        pts = decompose_segment(cur_x, cur_z, tx, tz, max_step)

        # For each intermediate point output set_position line + r.sleep()
        for i, (px, pz) in enumerate(pts):
            is_final = (i == len(pts) - 1)
            repeats = hold_draw if (mode == "DRAW" and is_final) else (hold_move if is_final else 1)
            for r_repeat in range(repeats):
                xs = format_f(px, prec)
                zs = format_f(pz, prec)
                ys = format_f(ty, prec)
                out_lines.append(f"set_position(x={xs}, y={ys}, z={zs}, frame_id='aruco_map')")
                out_lines.append("r.sleep()")
        cur_x, cur_z, cur_y = tx, tz, ty

    return out_lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', '-t', default='trajectory.txt')
    parser.add_argument('--config', '-c', default='config.json')
    parser.add_argument('--out', '-o', default=None, help="output file (overrides config)")
    args = parser.parse_args()

    traj_path = Path(args.traj)
    cfg_path = Path(args.config)
    if not traj_path.exists():
        print("Trajectory file not found:", traj_path)
        return
    if not cfg_path.exists():
        print("Config file not found:", cfg_path)
        return

    cfg = load_config(cfg_path)
    traj = parse_traj(traj_path)

    commands = generate_commands(traj, cfg)

    out_file = args.out if args.out else cfg.get("output_file_flyght", "flyght.py")
    out_path = Path(out_file)
    out_path.write_text("\n".join(commands), encoding='utf-8')
    print(f"Wrote {len(commands)} command lines to {out_path}")

if __name__ == '__main__':
    main()
