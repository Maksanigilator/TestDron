#!/usr/bin/env python3
# slicer2d_fixed.py
"""
2D-slicer для дрона-художника (исправленная версия):
- Обрабатывает бинарное изображение (OpenCV)
- Строит полигоны с отверстиями (OpenCV hierarchy -> Shapely)
- Для каждого полигона:
    * генерирует N периметров (pyclipper offset внутрь)
    * вычисляет внутреннюю область = полигон с отступом N * ширина_линии
    * генерирует заполнение только внутри внутренней области
- Сегментирует, упорядочивает (ближайший сосед) и экспортирует trajectory.txt

Usage:
  python slicer2d_fixed.py --config config.json

Dependencies:
  pip install numpy opencv-python shapely pyclipper
"""
import os
import json
import math
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Union

import numpy as np
import cv2

from shapely.geometry import Polygon, LineString, MultiLineString, MultiPolygon, Point
from shapely.ops import unary_union
import pyclipper

# -----------------------
# Константы и помощники
# -----------------------
SCALE_INT = 1000000  # масштаб метры -> целые для pyclipper (разрешение микрон)


def meters_to_ints(coords: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
    return [(int(round(x * SCALE_INT)), int(round(y * SCALE_INT))) for (x, y) in coords]


def ints_to_meters(coords: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
    return [(x / SCALE_INT, y / SCALE_INT) for (x, y) in coords]


def round_pt(p, nd=6):
    return (round(float(p[0]), nd), round(float(p[1]), nd))


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# -----------------------
# 1) Предобработка
# -----------------------
def preprocess_image(path: str, morph_open: int = 0, morph_close: int = 0) -> np.ndarray:
    """Загружает и предобрабатывает изображение"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")

    # Бинаризация
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Морфологические операции
    if morph_open > 0:
        kernel = np.ones((morph_open, morph_open), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    if morph_close > 0:
        kernel = np.ones((morph_close, morph_close), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    return bw


# -----------------------
# 2) Построение полигонов с отверстиями
# -----------------------
def contours_to_polygons(bw: np.ndarray, area_m: float = 2.0) -> Tuple[
    Union[Polygon, MultiPolygon], Tuple[float, float, float, float, float]]:
    """
    Находит контуры с помощью OpenCV (RETR_CCOMP) и собирает shapely полигоны (внешние с отверстиями).
    Возвращает (MultiPolygon_or_Polygon, map_info)
    map_info = (scale_m_per_px, cx, cy, bbox_w_px, bbox_h_px)
    Полигоны в метрах и центрированы в (0,0).
    """
    # Инвертируем для findContours (нужен белый передний план)
    inv = cv2.bitwise_not(bw)
    contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if not contours:
        raise RuntimeError("Контуры не найдены в изображении")

    # Вычисляем ограничивающий прямоугольник всех точек контуров
    all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
    x_min, y_min = np.min(all_pts, axis=0)
    x_max, y_max = np.max(all_pts, axis=0)

    w = x_max - x_min + 1
    h = y_max - y_min + 1
    scale = area_m / max(w, h)  # метры на пиксель
    cx = x_min + w / 2.0
    cy = y_min + h / 2.0

    # Строим маппинг: внешние контуры (parent == -1) -> список отверстий
    hier = hierarchy[0] if hierarchy is not None else []
    outer_to_holes = {}

    # Находим внешние контуры
    for i, hinfo in enumerate(hier):
        parent = int(hinfo[3])
        if parent == -1:
            outer_to_holes[i] = []

    # Для контуров с parent != -1, находим их родителя верхнего уровня
    for i, hinfo in enumerate(hier):
        parent = int(hinfo[3])
        if parent != -1:
            # Поднимаемся к самому верхнему предку
            ancestor = parent
            while int(hier[ancestor][3]) != -1:
                ancestor = int(hier[ancestor][3])

            if ancestor in outer_to_holes:
                outer_to_holes[ancestor].append(i)

    # Строим полигоны
    polygons = []
    min_contour_area = 50  # минимальная площадь контура в пикселях

    for outer_idx, hole_indices in outer_to_holes.items():
        try:
            # Проверяем минимальную площадь внешнего контура
            if cv2.contourArea(contours[outer_idx]) < min_contour_area:
                continue

            # Внешний контур
            ext_cnt = contours[outer_idx].reshape(-1, 2).astype(float)

            # Проверяем количество точек
            if len(ext_cnt) < 3:
                continue

            ext_coords = [((p[0] - cx) * scale, (cy - p[1]) * scale) for p in ext_cnt]

            # Убедимся, что у нас достаточно уникальных точек
            if len(set(ext_coords)) < 3:
                continue

            # Отверстия
            holes = []
            for hole_idx in hole_indices:
                hole_cnt = contours[hole_idx].reshape(-1, 2).astype(float)

                # Проверяем отверстие
                if len(hole_cnt) < 3 or cv2.contourArea(contours[hole_idx]) < min_contour_area:
                    continue

                hole_coords = [((p[0] - cx) * scale, (cy - p[1]) * scale) for p in hole_cnt]

                if len(set(hole_coords)) >= 3:
                    holes.append(hole_coords)

            # Создаем полигон
            poly = Polygon(ext_coords, holes=holes if holes else None)

            if poly.is_valid and not poly.is_empty and poly.area > 1e-8:
                polygons.append(poly)
            elif not poly.is_valid:
                # Пытаемся исправить невалидный полигон
                try:
                    poly_fixed = poly.buffer(0)
                    if poly_fixed.is_valid and not poly_fixed.is_empty and poly_fixed.area > 1e-8:
                        polygons.append(poly_fixed)
                except:
                    continue

        except Exception as e:
            print(f"Ошибка при создании полигона {outer_idx}: {e}")
            continue

    if not polygons:
        raise RuntimeError("Валидные полигоны не созданы")

    result = unary_union(polygons) if len(polygons) > 1 else polygons[0]
    map_info = (scale, cx, cy, w, h)
    return result, map_info


# -----------------------
# 3) Генератор периметров
# -----------------------
def generate_perimeters_for_polygon(poly: Polygon, num_perimeters: int, line_width: float, single_line: bool = False) -> List[
    List[Tuple[float, float]]]:
    """
    Возвращает список периметров для одного полигона (от внешнего к внутреннему).
    Использует pyclipper offset внутрь с дельтами (i+0.5)*line_width.
    Параметр single_line=True заставит функцию вернуть только одну центральную линию (смещение -0.5*line_width)
    """
    perimeters = []
    if poly.is_empty or not poly.is_valid:
        return perimeters

    try:
        # Подготавливаем целочисленные пути
        outer_path = meters_to_ints(list(poly.exterior.coords)[:-1])  # убираем дублирующую точку
        hole_paths = [meters_to_ints(list(h.coords)[:-1]) for h in poly.interiors]

        if single_line:
            # Вернуть только одну центральную линию вдоль внешнего периметра (без внутренних отступов)
            delta = -0.5 * line_width
            delta_int = int(round(delta * SCALE_INT))
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(outer_path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            try:
                solution = pco.Execute(delta_int)
                if solution:
                    # используем первый путь решения (может быть несколько при сложных полигонах)
                    path = solution[0]
                    if len(path) >= 3:
                        pts_m = ints_to_meters(path)
                        if pts_m[0] != pts_m[-1]:
                            pts_m.append(pts_m[0])
                        perimeters.append([round_pt(p, 6) for p in pts_m])
            except Exception as e:
                print(f"Ошибка при создании single-line периметра: {e}")
            return perimeters

        for i in range(num_perimeters):
            delta = -(i + 0.5) * line_width  # внутрь по центральной линии
            delta_int = int(round(delta * SCALE_INT))

            pco = pyclipper.PyclipperOffset()
            pco.AddPath(outer_path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

            # Добавляем отверстия
            for hole_path in hole_paths:
                pco.AddPath(hole_path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

            try:
                solution = pco.Execute(delta_int)
                if not solution:
                    break

                # solution может содержать несколько замкнутых путей
                for path in solution:
                    if len(path) >= 3:
                        pts_m = ints_to_meters(path)
                        # Замыкаем контур
                        if pts_m[0] != pts_m[-1]:
                            pts_m.append(pts_m[0])
                        perimeters.append([round_pt(p, 6) for p in pts_m])

            except Exception as e:
                print(f"Ошибка при создании периметра {i}: {e}")
                break

    except Exception as e:
        print(f"Ошибка при обработке полигона для периметров: {e}")

    return perimeters


# -----------------------
# 4) Вычисление внутренней области для заполнения
# -----------------------
def compute_infill_area(poly: Polygon, num_perimeters: int, line_width: float) -> Optional[
    Union[Polygon, MultiPolygon]]:
    """Вычисляет область для заполнения (полигон с отступом на ширину периметров)"""
    if not poly.is_valid or poly.is_empty:
        return None

    inset_dist = -float(num_perimeters) * line_width
    if abs(inset_dist) < 1e-8:
        return poly

    try:
        inner = poly.buffer(inset_dist, join_style=2)  # round joins
        if inner.is_empty:
            return None
        return inner
    except Exception as e:
        print(f"Ошибка при вычислении области заполнения: {e}")
        return None


# -----------------------
# 5) Генератор заполнения
# -----------------------
def generate_scanlines(bounds: Tuple[float, float, float, float], spacing: float, angle_deg: float = 0.0) -> List[
    LineString]:
    """
    Генерирует линии сканирования для заданной области
    bounds: (minx, miny, maxx, maxy)
    """
    minx, miny, maxx, maxy = bounds
    angle_rad = math.radians(angle_deg)

    # Увеличиваем область для покрытия после поворота
    diagonal = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2

    # Генерируем линии в локальной системе координат
    extend = diagonal
    y_range = int((2 * extend) / spacing) + 1
    lines = []

    for i in range(-y_range // 2, y_range // 2 + 1):
        y_local = i * spacing
        x1_local = -extend
        x2_local = extend

        # Поворачиваем и переносим в мировые координаты
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        x1 = center_x + (x1_local * cos_a - y_local * sin_a)
        y1 = center_y + (x1_local * sin_a + y_local * cos_a)
        x2 = center_x + (x2_local * cos_a - y_local * sin_a)
        y2 = center_y + (x2_local * sin_a + y_local * cos_a)

        lines.append(LineString([(x1, y1), (x2, y2)]))

    return lines


def generate_infill(poly: Union[Polygon, MultiPolygon], infill_type: str = "zigzag",
                    spacing: float = 0.05, angle: float = 0.0) -> List[LineString]:
    """
    Генерирует заполнение, обрезанное по полигону
    infill_type: 'zigzag'|'concentric'|'grid'|'triangular'
    """
    if poly is None or poly.is_empty:
        return []

    segments = []

    # Нормализуем к списку полигонов
    if isinstance(poly, Polygon):
        polygons = [poly]
    elif isinstance(poly, MultiPolygon):
        polygons = list(poly.geoms)
    else:
        return []

    for p in polygons:
        if not p.is_valid or p.is_empty:
            continue

        minx, miny, maxx, maxy = p.bounds

        if infill_type == "concentric":
            # Концентрические офсеты
            try:
                current_poly = p
                inset = -spacing
                while not current_poly.is_empty:
                    # Добавляем контур как LineString
                    if isinstance(current_poly, Polygon):
                        segments.append(LineString(current_poly.exterior.coords))
                        for interior in current_poly.interiors:
                            segments.append(LineString(interior.coords))
                    elif isinstance(current_poly, MultiPolygon):
                        for sub_poly in current_poly.geoms:
                            segments.append(LineString(sub_poly.exterior.coords))
                            for interior in sub_poly.interiors:
                                segments.append(LineString(interior.coords))

                    # Следующий офсет
                    current_poly = current_poly.buffer(inset, join_style=2)
            except Exception as e:
                print(f"Ошибка при генерации концентрического заполнения: {e}")
        else:
            # Линейные заполнения
            try:
                lines = generate_scanlines((minx, miny, maxx, maxy), spacing, angle)
                for line in lines:
                    intersection = p.intersection(line)
                    if intersection.is_empty:
                        continue

                    if isinstance(intersection, LineString):
                        segments.append(intersection)
                    elif isinstance(intersection, MultiLineString):
                        segments.extend(list(intersection.geoms))

                # Дополнительные направления для сетки и треугольного заполнения
                if infill_type == "grid":
                    lines2 = generate_scanlines((minx, miny, maxx, maxy), spacing, angle + 90.0)
                    for line in lines2:
                        intersection = p.intersection(line)
                        if intersection.is_empty:
                            continue
                        if isinstance(intersection, LineString):
                            segments.append(intersection)
                        elif isinstance(intersection, MultiLineString):
                            segments.extend(list(intersection.geoms))

                elif infill_type == "triangular":
                    for add_angle in [60.0, 120.0]:
                        lines_tri = generate_scanlines((minx, miny, maxx, maxy), spacing, angle + add_angle)
                        for line in lines_tri:
                            intersection = p.intersection(line)
                            if intersection.is_empty:
                                continue
                            if isinstance(intersection, LineString):
                                segments.append(intersection)
                            elif isinstance(intersection, MultiLineString):
                                segments.extend(list(intersection.geoms))

            except Exception as e:
                print(f"Ошибка при генерации линейного заполнения: {e}")

    return segments


# -----------------------
# 6) Сегментация / семплирование
# -----------------------
def linestring_to_samples(ls: LineString, spacing: float) -> List[Tuple[float, float]]:
    """Преобразует LineString в точки с заданным интервалом"""
    if ls.is_empty or ls.length < 1e-8:
        return []

    try:
        length = ls.length
        num_points = max(2, int(math.ceil(length / spacing)))
        points = []

        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            point = ls.interpolate(t, normalized=True)
            points.append((float(point.x), float(point.y)))

        return points
    except Exception as e:
        print(f"Ошибка при семплировании LineString: {e}")
        return []


def segmentize_segments(segments: List[LineString], spacing: float) -> List[List[Tuple[float, float]]]:
    """Преобразует список LineString в список точек"""
    result = []
    for segment in segments:
        points = linestring_to_samples(segment, spacing)
        if len(points) >= 2:
            result.append([round_pt(p, 6) for p in points])
    return result


# -----------------------
# 7) Упорядочивание (жадный ближайший сосед)
# -----------------------
def order_segments_nn(segments_pts: List[List[Tuple[float, float]]],
                      start: Tuple[float, float] = (0.0, 0.0)) -> List[List[Tuple[float, float]]]:
    """Упорядочивает сегменты по принципу ближайшего соседа"""
    if not segments_pts:
        return []

    remaining = segments_pts.copy()
    ordered = []
    current_pos = start

    while remaining:
        best_idx = 0
        best_dist = float('inf')
        best_reverse = False

        for i, segment in enumerate(remaining):
            if len(segment) < 2:
                continue

            # Проверяем расстояние до начала и конца сегмента
            start_dist = distance(current_pos, segment[0])
            end_dist = distance(current_pos, segment[-1])

            if start_dist < best_dist:
                best_dist = start_dist
                best_idx = i
                best_reverse = False

            if end_dist < best_dist:
                best_dist = end_dist
                best_idx = i
                best_reverse = True

        # Берем лучший сегмент
        segment = remaining.pop(best_idx)
        if best_reverse:
            segment = list(reversed(segment))

        ordered.append(segment)
        current_pos = segment[-1]

    return ordered


# -----------------------
# 8) Экспорт
# -----------------------
#def write_trajectory(output_path: str, segments_ordered: List[List[Tuple[float, float]]],
#                    z_draw: float, z_transit: float, speed: float = 0.5):
def write_trajectory(output_path: str, segments_ordered: List[List[Tuple[float, float]]], speed: float = 0.5):

    """Записывает траекторию в файл"""
    with open(output_path, 'w', encoding='utf-8') as f:
        # Начальная позиция
        f.write(f"MOVE 0.000000 0.000000 \n")  # Начальная позиция

        for segment in segments_ordered:
            if len(segment) < 2:
                continue

            # Переход к началу сегмента на высоте транзита
            x0, y0 = segment[0]
            f.write(f"MOVE {x0:.6f} {y0:.6f} \n")

            # Опускаемся для рисования (только Z координата)
            f.write(f"MOVE {x0:.6f} {y0:.6f} \n")

            # Рисуем сегмент - первая точка тоже DRAW
            for x, y in segment:
                f.write(f"DRAW {x:.6f} {y:.6f}\n")

            # Поднимаемся после рисования (только Z координата)
            x_end, y_end = segment[-1]
            f.write(f"MOVE {x_end:.6f} {y_end:.6f}\n")

        # Возврат в начальную позицию
        f.write(f"MOVE 0.000000 0.000000\n")
        f.write(f"MOVE 0.000000 0.000000\n")


# -----------------------
# Основной пайплайн
# -----------------------
def run_pipeline(config_path: str):
    """Основная функция обработки"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"Ошибка чтения конфига: {e}")
        return

    # Параметры из конфига
    input_image = config['input_image']
    output_file = config.get('output_file', 'trajectory.txt')
    area_m = float(config.get('area_m', 2.0))
    spacing_m = float(config.get('spacing_m', 0.02))  # изменил по умолчанию на 2 см
    line_width = float(config.get('line_width', 0.02))
    num_perimeters = int(config.get('num_perimeters', 2))
    # Новые опции:
    # infill_enabled: если False — полностью отключает генерацию заполнения
    infill_enabled = bool(config.get('infill_enabled', True))
    # single_line_perimeter: если True — генерирует только одну центральную линию периметра (смещение -0.5*line_width)
    single_line_perimeter = bool(config.get('single_line_perimeter', False))
    infill_type = config.get('infill_type', 'zigzag')
    infill_angle = float(config.get('infill_angle', 0.0))
    #z_draw = float(config.get('z_draw', 1.0))
    #z_transit = float(config.get('z_transit', 1.3))
    morph_open = int(config.get('morph_open', 0))
    morph_close = int(config.get('morph_close', 0))
    order_start = tuple(config.get('order_start', [0.0, 0.0]))
    speed = float(config.get('speed', 0.5))

    print("Предобработка изображения...")
    try:
        bw = preprocess_image(input_image, morph_open=morph_open, morph_close=morph_close)
    except Exception as e:
        print(f"Ошибка предобработки: {e}")
        return

    print("Построение полигонов...")
    try:
        poly_union, map_info = contours_to_polygons(bw, area_m=area_m)
        print(f"Полигоны готовы. Тип геометрии: {poly_union.geom_type}")
        print(f"Границы: {poly_union.bounds}")
        print(f"Масштаб: {map_info[0]:.6f} м/пиксель")
    except Exception as e:
        print(f"Ошибка построения полигонов: {e}")

        # Дополнительная диагностика
        print("Пытаемся диагностировать проблему...")
        inv = cv2.bitwise_not(bw)
        contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        print(f"Найдено контуров: {len(contours)}")
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            print(f"Контур {i}: точек={len(cnt)}, площадь={area:.1f}")
        return

    # Собираем все сегменты
    all_segments = []

    # Итерируемся по каждому полигону отдельно
    if isinstance(poly_union, Polygon):
        polygons = [poly_union]
    elif isinstance(poly_union, MultiPolygon):
        polygons = list(poly_union.geoms)
    else:
        print("Неподдерживаемый тип геометрии")
        return

    for idx, polygon in enumerate(polygons):
        if not polygon.is_valid or polygon.is_empty:
            continue

        print(f"[Полигон {idx}] площадь={polygon.area:.6f} м², границы={polygon.bounds}")

        # Генерируем периметры
        try:
            perimeters = generate_perimeters_for_polygon(polygon, num_perimeters, line_width, single_line=single_line_perimeter)
            print(f"[Полигон {idx}] периметров: {len(perimeters)}")

            for perimeter in perimeters:
                if len(perimeter) >= 2:
                    ls = LineString(perimeter)
                    points = linestring_to_samples(ls, spacing_m)
                    if len(points) >= 2:
                        all_segments.append([round_pt(pt, 6) for pt in points])
        except Exception as e:
            print(f"Ошибка генерации периметров для полигона {idx}: {e}")

        # Внутренняя область для заполнения (учитывается флаг infill_enabled)
        if infill_enabled:
            try:
                inner = compute_infill_area(polygon, num_perimeters, line_width)
                if inner is None or inner.is_empty:
                    print(f"[Полигон {idx}] внутренняя область слишком мала - пропускаем заполнение")
                    continue

                # Генерируем заполнение
                infill_lines = generate_infill(inner, infill_type=infill_type, spacing=spacing_m, angle=infill_angle)
                infill_segments = segmentize_segments(infill_lines, spacing_m)
                print(f"[Полигон {idx}] сегментов заполнения: {len(infill_segments)}")
                all_segments.extend(infill_segments)

            except Exception as e:
                print(f"Ошибка генерации заполнения для полигона {idx}: {e}")
        else:
            print(f"[Полигон {idx}] заполнение отключено (infill_enabled=False)")

    print(f"Всего сегментов: {len(all_segments)}")

    if not all_segments:
        print("Сегменты не сгенерированы - нечего записывать.")
        return

    print("Упорядочивание сегментов (ближайший сосед)...")
    try:
        ordered = order_segments_nn(all_segments, start=order_start)
    except Exception as e:
        print(f"Ошибка упорядочивания: {e}")
        return

    print(f"Запись траектории в {output_file}")
    try:
        #write_trajectory(output_file, ordered, z_draw=z_draw, z_transit=z_transit, speed=speed)
        write_trajectory(output_file, ordered, speed=speed)
        print(f"Готово. Траектория сохранена в {output_file}")
    except Exception as e:
        print(f"Ошибка записи файла: {e}")


# -----------------------
# CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="2D слайсер для дрона-художника")
    parser.add_argument('--config', '-c', help="Путь к config.json", default='config.json')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"Файл конфигурации не найден: {args.config}")
        print("Создайте config.json с необходимыми параметрами")
        return

    run_pipeline(args.config)


if __name__ == "__main__":
    main()