#!/usr/bin/env python3
"""
trajectory_to_commands.py

Reads trajectory.txt (lines: "MOVE x y" or "DRAW x y") and config.json,
decomposes movements so that set_position calls have step <= speed * dt,
supports separate draw_speed and move_speed, reinstates servo toggles,
writes a plain command file (sequence of pi.set_servo_pulsewidth(...) and
set_position(...) lines). After every set_position() a `r.sleep()` is emitted.

Usage:
  python trajectory_to_commands.py --traj trajectory.txt --config config.json

Config keys used (defaults shown):
  offset_x: 0.0
  offset_z: 0.0
  default_y / draw_y: 0.05       # Y used when drawing (DRAW)
  move_y: 0.20                   # Y used when moving/transit (MOVE)
  dt: 0.1
  max_speed: 5.0                 # safety upper bound if needed
  draw_speed: 1.5
  move_speed: 5.0
  servo_pin: 21
  servo_on: 2000
  servo_off: 1000
  hold_draw: 2
  hold_move: 1
  start_from_origin: false
  output_file_flight: "commands.txt"
  float_precision: 6

Output file will contain (example lines):
  import math
  import rospy
  r = rospy.Rate(10)
  pi.set_servo_pulsewidth(21, 2000)
  set_position(x=0.000000, y=0.050000, z=0.800000, frame_id='aruco_map', yaw = math.pi/2)
  r.sleep()
  ...
"""
import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Tuple

def load_config(path: Path) -> dict:
    txt = path.read_text(encoding='utf-8')
    # allow simple comments // and /* */
    txt = re.sub(r'//.*', '', txt)
    txt = re.sub(r'/\*.*?\*/', '', txt, flags=re.S)
    return json.loads(txt)

def parse_traj(path: Path) -> List[Tuple[str,float,float]]:
    pts = []
    with open(path, 'r', encoding='utf-8') as f:
        for ln in f:
            s = ln.strip()
            if not s or s.startswith('#'):
                continue
            parts = s.split()
            if len(parts) >= 3 and parts[0].upper() in ("MOVE","DRAW"):
                try:
                    mode = parts[0].upper()
                    x = float(parts[1]); y = float(parts[2])
                    pts.append((mode, x, y))
                    continue
                except:
                    pass
            # try to parse navigate_wait-like lines
            if "x=" in s and "y=" in s:
                try:
                    m = re.search(r"x\s*=\s*([-+]?\d*\.?\d+)", s)
                    n = re.search(r"y\s*=\s*([-+]?\d*\.?\d+)", s)
                    if m and n:
                        mode = "DRAW" if "draw" in s.lower() or "#draw" in s.lower() else "MOVE"
                        x = float(m.group(1)); y = float(n.group(1))
                        pts.append((mode, x, y))
                except:
                    pass
    return pts

def decompose_segment(cur_x: float, cur_z: float, tx: float, tz: float, step_max: float) -> List[Tuple[float,float]]:
    dx = tx - cur_x
    dz = tz - cur_z
    dist = math.hypot(dx, dz)
    if dist == 0.0:
        return [(tx, tz)]
    steps = int(math.ceil(dist / step_max))
    if steps < 1:
        steps = 1
    pts = []
    for s in range(1, steps + 1):
        t = s / steps
        xi = cur_x + dx * t
        zi = cur_z + dz * t
        pts.append((xi, zi))
    return pts

def fmt(x: float, prec:int) -> str:
    return f"{x:.{prec}f}"

def generate_commands(traj, cfg):
    offset_x = float(cfg.get("offset_x", 0.0))
    offset_z = float(cfg.get("offset_z", 0.0))
    draw_y = float(cfg.get("default_y", cfg.get("draw_y", 0.05)))
    move_y = float(cfg.get("move_y", cfg.get("move_y", 0.20)))
    dt = float(cfg.get("dt", 0.1))
    # speeds
    global_max_speed = float(cfg.get("max_speed", 5.0))
    draw_speed = float(cfg.get("draw_speed", cfg.get("max_speed", 1.5)))  # if draw_speed absent fallback sensibly
    move_speed = float(cfg.get("move_speed", cfg.get("max_speed", global_max_speed)))
    # clamp speeds to global_max_speed
    draw_speed = min(draw_speed, global_max_speed)
    move_speed = min(move_speed, global_max_speed)

    servo_pin = int(cfg.get("servo_pin", 21))
    servo_on = int(cfg.get("servo_on", 2000))
    servo_off = int(cfg.get("servo_off", 1000))
    hold_draw = int(cfg.get("hold_draw", 2))
    hold_move = int(cfg.get("hold_move", 1))
    start_from_origin = bool(cfg.get("start_from_origin", False))
    prec = int(cfg.get("float_precision", 6))

    # max step per dt depends on mode (we will compute per segment)
    # We'll produce command lines as strings
    out_lines: List[str] = []
    # header
    out_lines.append("import math")
    out_lines.append("import rospy")
    # compute rate from dt: safest to round to int
    try:
        hz = int(round(1.0 / dt))
        if hz <= 0: hz = 10
    except Exception:
        hz = 10
    out_lines.append(f"r = rospy.Rate({hz})")
    out_lines.append("")

    if not traj:
        return out_lines

    # prepare transformed targets: x_out = x_in + offset_x, z_out = y_in + offset_z
    targets = []
    for mode, xin, yin in traj:
        xout = xin + offset_x
        zout = yin + offset_z
        targets.append((mode, xout, zout))

    # initial current position
    if start_from_origin:
        cur_x = 0.0
        cur_z = 0.0
    else:
        cur_x = targets[0][1]
        cur_z = targets[0][2]

    servo_state = None  # "ON" or "OFF", track last state to avoid duplicate commands

    for idx, (mode, tx, tz) in enumerate(targets):
        # decide desired servo state for this segment
        desired_servo_on = (mode == "DRAW")
        # toggle servo if needed BEFORE starting movement to target (so when drawing begins servo already ON)
        if servo_state is None:
            # initial: set to desired state
            if desired_servo_on:
                out_lines.append(f"pi.set_servo_pulsewidth({servo_pin}, {servo_on})")
                servo_state = "ON"
            else:
                out_lines.append(f"pi.set_servo_pulsewidth({servo_pin}, {servo_off})")
                servo_state = "OFF"
        else:
            # If state change needed, append toggle prior to set_position calls
            if desired_servo_on and servo_state != "ON":
                out_lines.append(f"pi.set_servo_pulsewidth({servo_pin}, {servo_on})")
                servo_state = "ON"
            elif (not desired_servo_on) and servo_state != "OFF":
                out_lines.append(f"pi.set_servo_pulsewidth({servo_pin}, {servo_off})")
                servo_state = "OFF"

        # choose speed and y depending on mode
        if mode == "DRAW":
            speed = draw_speed
            y_for_cmd = draw_y
        else:
            speed = move_speed
            y_for_cmd = move_y

        # compute maximum allowed step length for this mode
        step_max = max(1e-9, speed * dt)

        # decompose movement from current to target using step_max
        pts = decompose_segment(cur_x, cur_z, tx, tz, step_max)

        # write set_position lines for each intermediate point.
        for i, (px, pz) in enumerate(pts):
            is_final = (i == len(pts) - 1)
            # choose repeats: for final point repeat hold_draw if DRAW else hold_move (simulate dwell)
            repeats = (hold_draw if (mode == "DRAW" and is_final) else (hold_move if is_final else 1))
            for rep in range(repeats):
                xs = fmt(px, prec)
                ys = fmt(y_for_cmd, prec)
                zs = fmt(pz, prec)
                # include frame_id and yaw exactly as requested
                out_lines.append(f"set_position(x={xs}, y={ys}, z={zs}, frame_id='aruco_map', yaw = math.pi/2)")
                out_lines.append("r.sleep()")
        # update current
        cur_x, cur_z = tx, tz

    return out_lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj', '-t', default='trajectory.txt')
    parser.add_argument('--config', '-c', default='config.json')
    parser.add_argument('--out', '-o', default=None, help="output file (overrides config output_file_flight)")
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

    out_file = args.out if args.out else cfg.get("output_file_flight", cfg.get("output_file", "commands.txt"))
    out_path = Path(out_file)
    out_path.write_text("\n".join(commands), encoding='utf-8')
    print(f"Wrote {len(commands)} lines to {out_path}")

if __name__ == '__main__':
    main()
