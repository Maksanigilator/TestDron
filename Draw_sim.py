#!/usr/bin/env python3
"""
visualize_trajectory.py

Визуализация траекторий дронов (gif + html slider + кадры).
Поддерживаемые входные форматы (в одном .txt файле):
  1) "MODE x y z"  (MODE = MOVE|DRAW)
  2) "navigate_wait(x=..., y=..., z=..., speed=..., frame_id='...')  #draw"  или без #draw
     - если в строке есть "draw" в комментарии, считается DRAW, иначе MOVE

Запуск:
  python visualize_trajectory.py trajectory.txt --outdir out_vis

Зависимости:
  pip install numpy matplotlib imageio opencv-python
"""

import os
import re
import argparse
from pathlib import Path
from typing import List, Dict
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import imageio.v2 as imageio

# ---------- Парсер траектории ----------
def parse_trajectory_file(path: Path) -> List[Dict]:
    """
    Возвращает список точек: [{"mode":"DRAW"|"MOVE","x":float,"y":float,"z":float}, ...]
    Поддерживает два формата (см. docstring).
    """
    points = []
    nav_re = re.compile(r"x\s*=\s*([-+]?\d*\.?\d+),\s*y\s*=\s*([-+]?\d*\.?\d+),\s*z\s*=\s*([-+]?\d*\.?\d+)")
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            parts = s.split()
            # Формат "MODE x y z"
            if len(parts) >= 4 and parts[0].upper() in ("MOVE", "DRAW"):
                try:
                    mode = parts[0].upper()
                    x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                    points.append({"mode": mode, "x": x, "y": y, "z": z})
                    continue
                except:
                    pass
            # Формат navigate_wait(...)
            m = nav_re.search(s)
            if m:
                x = float(m.group(1)); y = float(m.group(2)); z = float(m.group(3))
                mode = "DRAW" if ("draw" in s.lower() or "#draw" in s.lower()) else "MOVE"
                points.append({"mode": mode, "x": x, "y": y, "z": z})
                continue
            # Попытка парсинга похожих строк (комментарии, CSV и т.д.)
            # Игнорируем нераспознанные строки
    return points

# ---------- Класс Drone ----------
class DroneTrajectory:
    def __init__(self, drone_id: str, points: List[Dict]):
        self.id = drone_id
        self.points = points  # list of {"mode","x","y","z"}
        # Precompute arrays
        self.xs = np.array([p["x"] for p in points]) if points else np.array([])
        self.ys = np.array([p["y"] for p in points]) if points else np.array([])
        self.zs = np.array([p["z"] for p in points]) if points else np.array([])
        self.modes = [p["mode"] for p in points]

    def __len__(self):
        return len(self.points)

# ---------- Визуализатор ----------
class TrajectoryVisualizer:
    def __init__(self, drones: List[DroneTrajectory], outdir: Path,
                 safety_radius: float = 0.30, fps: int = 6, dpi: int = 150):
        self.drones = drones
        self.outdir = outdir
        self.frames_dir = outdir / "frames"
        self.gif_path = outdir / "trajectory_anim.gif"
        self.html_path = outdir / "trajectory_slider.html"
        self.safety_radius = safety_radius
        self.fps = fps
        self.dpi = dpi
        self.colors = plt.get_cmap('tab10').colors
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def compute_bounds(self, pad=0.2):
        # Собираем все x,y
        xs_all, ys_all = [], []
        for d in self.drones:
            if len(d) > 0:
                xs_all.extend(d.xs.tolist())
                ys_all.extend(d.ys.tolist())
        if not xs_all:
            # дефолтные границы
            return (-1.0-pad, 1.0+pad, -1.0-pad, 1.0+pad)
        xmin = min(xs_all) - pad
        xmax = max(xs_all) + pad
        ymin = min(ys_all) - pad
        ymax = max(ys_all) + pad
        # Если область очень мала, скорректируем до минимум 2x2 метров (если нужно)
        """
        if xmax - xmin < 2.0:
            cx = (xmax + xmin) / 2.0
            xmin = cx - 1.0; xmax = cx + 1.0
        if ymax - ymin < 2.0:
            cy = (ymax + ymin) / 2.0
            ymin = cy - 1.0; ymax = cy + 1.0
        """
        return xmin, xmax, ymin, ymax

    def render(self):
        # Максимальное количество шагов (frames) — возьмём максимум длины траекторий
        max_steps = max((len(d) for d in self.drones), default=0)
        if max_steps == 0:
            raise RuntimeError("Нет точек для визуализации в ни в одном файле.")

        xmin, xmax, ymin, ymax = self.compute_bounds()
        fig, ax = plt.subplots(figsize=(6,6))
        ax.set_facecolor("white")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal', 'box')
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title("Drone trajectories (projection on wall/canvas)")

        # Prepare artists for each drone: safety circle, scatter (current pos), draw_line, move_line
        artists = []
        for idx, drone in enumerate(self.drones):
            color = self.colors[idx % len(self.colors)]
            # safety circle initial at origin (will update)
            circ = Circle((0,0), self.safety_radius, alpha=0.15, facecolor=color, edgecolor='black', lw=0.5)
            ax.add_patch(circ)
            draw_line, = ax.plot([], [], linewidth=2.0, color='black')  # DRAW solid
            move_line, = ax.plot([], [], linestyle='--', linewidth=1.0, alpha=0.6, color='gray')  # MOVE dashed
            scatter = ax.scatter([], [], s=120, zorder=5, marker='o', edgecolors='black', facecolors=color)
            artists.append({
                "circle": circ, "draw_line": draw_line, "move_line": move_line,
                "scatter": scatter, "color": color,
                "acc_draw_x": [], "acc_draw_y": [], "acc_move_x": [], "acc_move_y": []
            })

        frame_files = []
        # Build frames: at step t we show for every drone all points up to index t (or last if shorter)
        for t in range(max_steps):
            for di, drone in enumerate(self.drones):
                art = artists[di]
                if len(drone) == 0:
                    # nothing to show
                    continue
                # choose index to sample: min(t, len-1)
                idx = min(t, len(drone)-1)
                mode = drone.modes[idx]
                x = float(drone.xs[idx]); y = float(drone.ys[idx])
                # update safety circle center
                art["circle"].center = (x,y)
                # append to accumulators based on mode
                if mode == "DRAW":
                    art["acc_draw_x"].append(x); art["acc_draw_y"].append(y)
                else:
                    art["acc_move_x"].append(x); art["acc_move_y"].append(y)
                # update lines and scatter
                art["draw_line"].set_data(art["acc_draw_x"], art["acc_draw_y"])
                art["move_line"].set_data(art["acc_move_x"], art["acc_move_y"])
                art["scatter"].set_offsets([[x,y]])
                # visual style: opaque when DRAW, translucent for MOVE (by checking current mode)
                if mode == "DRAW":
                    art["scatter"].set_alpha(1.0)
                else:
                    art["scatter"].set_alpha(0.45)

            # Save frame
            fname = self.frames_dir / f"frame_{t:04d}.png"
            plt.savefig(fname, dpi=self.dpi, bbox_inches='tight')
            frame_files.append(fname.name)

        plt.close(fig)

        # Create GIF
        imgs = [imageio.imread(str(self.frames_dir / fn)) for fn in frame_files]
        imageio.mimsave(str(self.gif_path), imgs, fps=self.fps)

        # Create simple HTML slider
        n = len(frame_files)
        html_lines = [
            "<!doctype html>",
            "<html><head><meta charset='utf-8'><title>Trajectory slider</title></head><body>",
            "<h3>Trajectory debug slider</h3>",
            f"<img id='frame' src='frames/{frame_files[0]}' style='max-width:90%;border:1px solid #ccc'/>",
            "<br/>",
            f"<input type='range' min='0' max='{n-1}' value='0' id='slider' style='width:80%'>",
            "<script>",
            "const slider = document.getElementById('slider');",
            "const img = document.getElementById('frame');",
            "slider.addEventListener('input', e => {",
            "  const idx = e.target.value; img.src = 'frames/frame_' + String(idx).padStart(4,'0') + '.png';",
            "});",
            "</script>",
            "<p>Solid line = DRAW, dashed = MOVE. Circle = safety radius.</p>",
            "</body></html>"
        ]
        with open(self.html_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(html_lines))

        return {"gif": str(self.gif_path), "frames_dir": str(self.frames_dir), "html": str(self.html_path)}

# ---------- CLI ----------
def collect_drones_from_path(path: Path) -> List[DroneTrajectory]:
    drones = []
    if path.is_dir():
        # все .txt файлы в папке — отдельные дроны
        for p in sorted(path.glob("*.txt")):
            pts = parse_trajectory_file(p)
            drones.append(DroneTrajectory(p.stem, pts))
    else:
        # единичный файл — один дрон
        pts = parse_trajectory_file(path)
        drones.append(DroneTrajectory(path.stem, pts))
    return drones

def main():
    parser = argparse.ArgumentParser(description="Visualize trajectory(ies) from trajectory.txt_drone_01_black.txt")
    parser.add_argument('input', nargs='?', help="Path to trajectory.txt or directory with .txt files", default="trajectory.txt_drone_01_black.txt")
    parser.add_argument('--indir', help="alternative: specify directory with many trajectory .txt files")
    parser.add_argument('--outdir', help="output directory", default="viz_output")
    parser.add_argument('--safety', type=float, default=0.30, help="safety radius (m)")
    parser.add_argument('--fps', type=int, default=6, help="gif frames per second")
    args = parser.parse_args()

    input_path = Path(args.indir) if args.indir else Path(args.input)
    if not input_path.exists():
        print("Входной путь не существует:", input_path)
        return

    drones = collect_drones_from_path(input_path)
    if not drones:
        print("Не найдено файлов с траекториями.")
        return

    viz = TrajectoryVisualizer(drones, Path(args.outdir), safety_radius=args.safety, fps=args.fps)
    out = viz.render()
    print("Готово. Результаты сохранены в", args.outdir)
    print("GIF:", out["gif"])
    print("Frames dir:", out["frames_dir"])
    print("HTML slider:", out["html"])

if __name__ == "__main__":
    main()
