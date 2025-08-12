#!/usr/bin/env python3
"""
slicing_engine.py - Ядро нарезчика для дронов-художников
Генерирует периметры и заполнение для полигонов
"""
import math
from typing import List, Tuple, Optional, Union, Dict, Any
from enum import Enum
import logging

from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, Point
from shapely.ops import unary_union
import pyclipper

logger = logging.getLogger(__name__)

# Константы
SCALE_FACTOR = 1000000  # Масштабирование для pyclipper (микронная точность)


class InfillType(Enum):
    """Типы заполнения"""
    LINES = "lines"  # Прямые линии
    ZIGZAG = "zigzag"  # Зигзаг (непрерывные линии)
    GRID = "grid"  # Сетка (два направления)
    TRIANGULAR = "triangular"  # Треугольная сетка
    CONCENTRIC = "concentric"  # Концентрические контуры
    HILBERT = "hilbert"  # Кривая Гильберта
    OCTAGRAM = "octagram"  # Восьмиугольные спирали


class SlicingEngine:
    """Ядро для генерации траекторий рисования"""

    def __init__(self):
        self.scale_factor = SCALE_FACTOR

    def _to_clipper_coords(self, coords: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Конвертирует координаты в целые числа для pyclipper"""
        return [(int(round(x * self.scale_factor)), int(round(y * self.scale_factor))) for x, y in coords]

    def _from_clipper_coords(self, coords: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """Конвертирует координаты из pyclipper обратно в float"""
        return [(x / self.scale_factor, y / self.scale_factor) for x, y in coords]

    def generate_perimeters(self,
                            polygon: Union[Polygon, MultiPolygon],
                            line_width: float,
                            num_perimeters: int,
                            start_from_outside: bool = True) -> List[List[Tuple[float, float]]]:
        """
        Генерирует периметры для полигона/мультиполигона

        Args:
            polygon: Исходный полигон
            line_width: Ширина линии
            num_perimeters: Количество периметров
            start_from_outside: Начинать с внешнего периметра

        Returns:
            Список периметров как списков координат
        """
        logger.info(f"Генерация {num_perimeters} периметров, ширина линии: {line_width}")

        perimeters = []

        # Нормализуем к списку полигонов
        if isinstance(polygon, Polygon):
            polygons = [polygon]
        elif isinstance(polygon, MultiPolygon):
            polygons = list(polygon.geoms)
        else:
            return perimeters

        for poly in polygons:
            if not poly.is_valid or poly.is_empty:
                continue

            poly_perimeters = self._generate_perimeters_for_polygon(
                poly, line_width, num_perimeters, start_from_outside
            )
            perimeters.extend(poly_perimeters)

        logger.info(f"Сгенерировано периметров: {len(perimeters)}")
        return perimeters

    def _generate_perimeters_for_polygon(self,
                                         polygon: Polygon,
                                         line_width: float,
                                         num_perimeters: int,
                                         start_from_outside: bool) -> List[List[Tuple[float, float]]]:
        """Генерирует периметры для одного полигона"""
        perimeters = []

        try:
            # Подготавливаем пути для pyclipper
            exterior_path = self._to_clipper_coords(list(polygon.exterior.coords)[:-1])
            hole_paths = [self._to_clipper_coords(list(hole.coords)[:-1]) for hole in polygon.interiors]

            for i in range(num_perimeters):
                # Вычисляем смещение
                offset_distance = -(i + 0.5) * line_width  # Внутрь, по центру линии
                offset_int = int(round(offset_distance * self.scale_factor))

                # Создаем offsetter
                pco = pyclipper.PyclipperOffset()
                pco.AddPath(exterior_path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

                # Добавляем отверстия
                for hole_path in hole_paths:
                    pco.AddPath(hole_path, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

                try:
                    solution = pco.Execute(offset_int)

                    if not solution:
                        break  # Больше нет места для периметров

                    # Обрабатываем результат
                    for path in solution:
                        if len(path) >= 3:
                            coords = self._from_clipper_coords(path)
                            # Замыкаем контур
                            if coords[0] != coords[-1]:
                                coords.append(coords[0])
                            perimeters.append(coords)

                except Exception as e:
                    logger.warning(f"Ошибка генерации периметра {i}: {e}")
                    break

        except Exception as e:
            logger.error(f"Ошибка обработки полигона для периметров: {e}")

        # Сортируем периметры (от внешнего к внутреннему или наоборот)
        if not start_from_outside:
            perimeters.reverse()

        return perimeters

    def compute_infill_area(self,
                            polygon: Union[Polygon, MultiPolygon],
                            line_width: float,
                            num_perimeters: int) -> Optional[Union[Polygon, MultiPolygon]]:
        """
        Вычисляет область для заполнения (с учетом периметров)

        Args:
            polygon: Исходный полигон
            line_width: Ширина линии
            num_perimeters: Количество периметров

        Returns:
            Область для заполнения или None
        """
        if not polygon or polygon.is_empty:
            return None

        # Вычисляем смещение внутрь на толщину всех периметров
        inset_distance = -float(num_perimeters) * line_width

        if abs(inset_distance) < 1e-8:
            return polygon

        try:
            inner_area = polygon.buffer(inset_distance, join_style=2)  # Round joins

            if inner_area.is_empty:
                logger.info("Область заполнения пуста после смещения")
                return None

            logger.info(f"Площадь заполнения: {inner_area.area:.6f} м²")
            return inner_area

        except Exception as e:
            logger.error(f"Ошибка вычисления области заполнения: {e}")
            return None

    def generate_infill(self,
                        area: Union[Polygon, MultiPolygon],
                        infill_type: InfillType,
                        line_spacing: float,
                        angle_deg: float = 0.0,
                        density: float = 1.0) -> List[List[Tuple[float, float]]]:
        """
        Генерирует заполнение для заданной области

        Args:
            area: Область для заполнения
            infill_type: Тип заполнения
            line_spacing: Расстояние между линиями
            angle_deg: Угол поворота в градусах
            density: Плотность заполнения (0.0-1.0)

        Returns:
            Список линий заполнения
        """
        if not area or area.is_empty:
            return []

        # Корректируем spacing с учетом плотности
        actual_spacing = line_spacing / max(0.1, density)

        logger.info(f"Генерация заполнения: тип={infill_type.value}, spacing={actual_spacing:.3f}, угол={angle_deg}°")

        # Маршрутизация по типу заполнения
        if infill_type == InfillType.CONCENTRIC:
            return self._generate_concentric_infill(area, actual_spacing)
        elif infill_type == InfillType.HILBERT:
            return self._generate_hilbert_infill(area, actual_spacing)
        else:
            return self._generate_line_infill(area, infill_type, actual_spacing, angle_deg)

    def _generate_line_infill(self,
                              area: Union[Polygon, MultiPolygon],
                              infill_type: InfillType,
                              spacing: float,
                              angle_deg: float) -> List[List[Tuple[float, float]]]:
        """Генерирует линейное заполнение"""
        infill_lines = []

        # Нормализуем к списку полигонов
        if isinstance(area, Polygon):
            polygons = [area]
        elif isinstance(area, MultiPolygon):
            polygons = list(area.geoms)
        else:
            return infill_lines

        angles = [angle_deg]

        # Определяем дополнительные углы для сложных типов заполнения
        if infill_type == InfillType.GRID:
            angles.append(angle_deg + 90.0)
        elif infill_type == InfillType.TRIANGULAR:
            angles.extend([angle_deg + 60.0, angle_deg + 120.0])
        elif infill_type == InfillType.OCTAGRAM:
            angles.extend([angle_deg + 45.0, angle_deg + 90.0, angle_deg + 135.0])

        for poly in polygons:
            for angle in angles:
                lines = self._generate_scanlines_for_polygon(poly, spacing, angle)

                if infill_type == InfillType.ZIGZAG:
                    # Соединяем линии в зигзаг
                    lines = self._connect_lines_to_zigzag(lines)

                infill_lines.extend(lines)

        return infill_lines

    def _generate_scanlines_for_polygon(self,
                                        polygon: Polygon,
                                        spacing: float,
                                        angle_deg: float) -> List[List[Tuple[float, float]]]:
        """Генерирует линии сканирования для полигона"""
        if polygon.is_empty:
            return []

        minx, miny, maxx, maxy = polygon.bounds
        angle_rad = math.radians(angle_deg)

        # Увеличиваем область для покрытия после поворота
        diagonal = math.sqrt((maxx - minx) ** 2 + (maxy - miny) ** 2)
        center_x = (minx + maxx) / 2
        center_y = (miny + maxy) / 2

        # Генерируем линии
        extend = diagonal
        num_lines = int((2 * extend) / spacing) + 1
        lines = []

        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        for i in range(-num_lines // 2, num_lines // 2 + 1):
            y_local = i * spacing

            # Создаем линию в локальных координатах
            x1_local, x2_local = -extend, extend

            # Поворачиваем и переносим
            x1 = center_x + (x1_local * cos_a - y_local * sin_a)
            y1 = center_y + (x1_local * sin_a + y_local * cos_a)
            x2 = center_x + (x2_local * cos_a - y_local * sin_a)
            y2 = center_y + (x2_local * sin_a + y_local * cos_a)

            scan_line = LineString([(x1, y1), (x2, y2)])

            # Пересекаем с полигоном
            try:
                intersection = polygon.intersection(scan_line)

                if intersection.is_empty:
                    continue

                if isinstance(intersection, LineString):
                    coords = list(intersection.coords)
                    if len(coords) >= 2:
                        lines.append(coords)
                elif isinstance(intersection, MultiLineString):
                    for line in intersection.geoms:
                        coords = list(line.coords)
                        if len(coords) >= 2:
                            lines.append(coords)

            except Exception as e:
                logger.debug(f"Ошибка пересечения линии сканирования: {e}")
                continue

        return lines

    def _connect_lines_to_zigzag(self, lines: List[List[Tuple[float, float]]]) -> List[List[Tuple[float, float]]]:
        """Соединяет отдельные линии в зигзаг-траектории"""
        if not lines:
            return []

        # Сортируем линии по Y координате их центров
        lines_with_centers = []
        for line in lines:
            if len(line) >= 2:
                center_y = sum(p[1] for p in line) / len(line)
                lines_with_centers.append((center_y, line))

        lines_with_centers.sort(key=lambda x: x[0])
        sorted_lines = [line for _, line in lines_with_centers]

        # Соединяем линии, реверсируя каждую вторую
        zigzag_lines = []
        current_zigzag = []

        for i, line in enumerate(sorted_lines):
            if i % 2 == 1:
                line = list(reversed(line))

            if not current_zigzag:
                current_zigzag = line[:]
            else:
                # Проверяем, можно ли соединить
                last_point = current_zigzag[-1]
                first_point = line[0]
                distance = math.sqrt((last_point[0] - first_point[0]) ** 2 + (last_point[1] - first_point[1]) ** 2)

                # Если линии близко, соединяем их
                if distance < 0.1:  # 10 см максимальный разрыв
                    current_zigzag.extend(line[1:])  # Исключаем первую точку чтобы избежать дубликата
                else:
                    # Завершаем текущий зигзаг и начинаем новый
                    if len(current_zigzag) >= 2:
                        zigzag_lines.append(current_zigzag)
                    current_zigzag = line[:]

        # Добавляем последний зигзаг
        if len(current_zigzag) >= 2:
            zigzag_lines.append(current_zigzag)

        return zigzag_lines

    def _generate_concentric_infill(self,
                                    area: Union[Polygon, MultiPolygon],
                                    spacing: float) -> List[List[Tuple[float, float]]]:
        """Генерирует концентрическое заполнение"""
        infill_lines = []

        if isinstance(area, Polygon):
            polygons = [area]
        elif isinstance(area, MultiPolygon):
            polygons = list(area.geoms)
        else:
            return infill_lines

        for poly in polygons:
            if poly.is_empty:
                continue

            try:
                current_poly = poly
                inset = -spacing

                while not current_poly.is_empty:
                    # Добавляем контур как траекторию
                    if isinstance(current_poly, Polygon):
                        exterior_coords = list(current_poly.exterior.coords)
                        if len(exterior_coords) >= 2:
                            infill_lines.append(exterior_coords)

                        # Добавляем отверстия
                        for interior in current_poly.interiors:
                            interior_coords = list(interior.coords)
                            if len(interior_coords) >= 2:
                                infill_lines.append(interior_coords)

                    elif isinstance(current_poly, MultiPolygon):
                        for sub_poly in current_poly.geoms:
                            exterior_coords = list(sub_poly.exterior.coords)
                            if len(exterior_coords) >= 2:
                                infill_lines.append(exterior_coords)

                            for interior in sub_poly.interiors:
                                interior_coords = list(interior.coords)
                                if len(interior_coords) >= 2:
                                    infill_lines.append(interior_coords)

                    # Следующий концентрический контур
                    current_poly = current_poly.buffer(inset, join_style=2)

            except Exception as e:
                logger.warning(f"Ошибка генерации концентрического заполнения: {e}")
                break

        return infill_lines

    def _generate_hilbert_infill(self,
                                 area: Union[Polygon, MultiPolygon],
                                 spacing: float) -> List[List[Tuple[float, float]]]:
        """Генерирует заполнение кривой Гильберта (упрощенная версия)"""
        # Для простоты реализуем как спиральное заполнение
        # В будущем можно добавить настоящую кривую Гильберта
        return self._generate_spiral_infill(area, spacing)

    def _generate_spiral_infill(self,
                                area: Union[Polygon, MultiPolygon],
                                spacing: float) -> List[List[Tuple[float, float]]]:
        """Генерирует спиральное заполнение"""
        infill_lines = []

        if isinstance(area, Polygon):
            polygons = [area]
        elif isinstance(area, MultiPolygon):
            polygons = list(area.geoms)
        else:
            return infill_lines

        for poly in polygons:
            if poly.is_empty:
                continue

            try:
                # Находим центр полигона
                centroid = poly.centroid
                center_x, center_y = centroid.x, centroid.y

                # Генерируем спираль от центра
                spiral_points = []
                max_radius = math.sqrt(poly.area / math.pi) * 1.5  # Приблизительный радиус покрытия

                angle = 0.0
                radius = 0.0
                angle_step = math.radians(10)  # 10 градусов на шаг

                while radius < max_radius:
                    x = center_x + radius * math.cos(angle)
                    y = center_y + radius * math.sin(angle)
                    point = Point(x, y)

                    if poly.contains(point):
                        spiral_points.append((x, y))

                    angle += angle_step
                    radius += spacing / (2 * math.pi)  # Увеличиваем радиус для поддержания spacing

                if len(spiral_points) >= 2:
                    infill_lines.append(spiral_points)

            except Exception as e:
                logger.warning(f"Ошибка генерации спирального заполнения: {e}")
                continue

        return infill_lines


# Пример использования
if __name__ == "__main__":
    import logging
    from shapely.geometry import Polygon

    logging.basicConfig(level=logging.INFO)

    # Создаем тестовый полигон
    test_polygon = Polygon([(0, 0), (2, 0), (2, 2), (0, 2), (0, 0)])

    engine = SlicingEngine()

    # Генерируем периметры
    perimeters = engine.generate_perimeters(
        test_polygon,
        line_width=0.02,
        num_perimeters=3
    )
    print(f"Периметров: {len(perimeters)}")

    # Вычисляем область заполнения
    infill_area = engine.compute_infill_area(
        test_polygon,
        line_width=0.02,
        num_perimeters=3
    )

    if infill_area:
        # Генерируем заполнение
        infill = engine.generate_infill(
            infill_area,
            InfillType.ZIGZAG,
            line_spacing=0.05,
            angle_deg=45.0
        )
        print(f"Линий заполнения: {len(infill)}")
    else:
        print("Область заполнения пуста")