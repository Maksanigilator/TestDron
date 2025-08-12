#!/usr/bin/env python3
"""
image_processor.py - Обработчик изображений для дронов-художников
Конвертирует растровые изображения в векторные данные
"""
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Класс для обработки изображений и конвертации в векторы"""

    def __init__(self):
        self.scale_factor = 1.0
        self.center_offset = (0.0, 0.0)

    def load_and_preprocess(self,
                            image_path: Union[str, Path],
                            target_size_m: float = 2.0,
                            morph_open: int = 0,
                            morph_close: int = 0,
                            invert: bool = False) -> np.ndarray:
        """
        Загружает и предобрабатывает изображение

        Args:
            image_path: Путь к изображению
            target_size_m: Размер наибольшей стороны в метрах
            morph_open: Размер ядра для морфологического открытия
            morph_close: Размер ядра для морфологического закрытия
            invert: Инвертировать изображение

        Returns:
            Бинарное изображение
        """
        logger.info(f"Загрузка изображения: {image_path}")

        # Загрузка изображения
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")

        logger.info(f"Размер изображения: {img.shape}")

        # Бинаризация
        if invert:
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        else:
            _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Морфологические операции
        if morph_open > 0:
            kernel = np.ones((morph_open, morph_open), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            logger.info(f"Применено морфологическое открытие: {morph_open}x{morph_open}")

        if morph_close > 0:
            kernel = np.ones((morph_close, morph_close), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            logger.info(f"Применено морфологическое закрытие: {morph_close}x{morph_close}")

        # Обрезка по контенту и масштабирование
        binary = self._crop_to_content(binary, target_size_m)

        return binary

    def _crop_to_content(self, binary: np.ndarray, target_size_m: float) -> np.ndarray:
        """Обрезает изображение по содержимому и вычисляет масштаб"""

        # Находим границы содержимого
        coords = np.column_stack(np.where(binary > 0))
        if len(coords) == 0:
            raise ValueError("Изображение не содержит объектов для обрезки")

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Добавляем небольшой отступ
        margin = 5
        y_min = max(0, y_min - margin)
        x_min = max(0, x_min - margin)
        y_max = min(binary.shape[0], y_max + margin)
        x_max = min(binary.shape[1], x_max + margin)

        # Обрезаем
        cropped = binary[y_min:y_max, x_min:x_max]

        # Делаем квадратным (добавляем padding)
        h, w = cropped.shape
        size = max(h, w)

        # Создаем квадратное изображение
        square = np.zeros((size, size), dtype=np.uint8)
        y_offset = (size - h) // 2
        x_offset = (size - w) // 2
        square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

        # Вычисляем масштаб: target_size_m / size_px
        self.scale_factor = target_size_m / size

        # Вычисляем смещение центра (для конвертации в координаты с центром в (0,0))
        self.center_offset = (size / 2.0, size / 2.0)

        logger.info(f"Обрезано до размера: {square.shape}")
        logger.info(f"Масштаб: {self.scale_factor:.6f} м/пикс")

        return square

    def extract_contours(self, binary: np.ndarray, min_area_px: int = 50) -> List[Tuple[np.ndarray, bool]]:
        """
        Извлекает контуры из бинарного изображения

        Args:
            binary: Бинарное изображение
            min_area_px: Минимальная площадь контура в пикселях

        Returns:
            List of (contour_points, is_hole) tuples
        """
        logger.info("Извлечение контуров...")

        # Для findContours нужен белый передний план на черном фоне
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            logger.warning("Контуры не найдены")
            return []

        logger.info(f"Найдено контуров: {len(contours)}")

        # Фильтруем по площади и определяем, что является отверстием
        filtered_contours = []
        hier = hierarchy[0] if hierarchy is not None else []

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area_px:
                continue

            # Определяем, является ли контур отверстием
            is_hole = False
            if len(hier) > i:
                parent = hier[i][3]
                is_hole = parent != -1

            # Конвертируем в координаты с центром в (0,0) и в метры
            points = contour.reshape(-1, 2).astype(float)
            # Переводим из пиксельных координат в метры с центром в (0,0)
            points_m = []
            for x, y in points:
                # Сдвигаем к центру и переводим в метры
                x_m = (x - self.center_offset[0]) * self.scale_factor
                y_m = (self.center_offset[1] - y) * self.scale_factor  # Инвертируем Y
                points_m.append([x_m, y_m])

            filtered_contours.append((np.array(points_m), is_hole))
            logger.debug(f"Контур {i}: площадь={area:.1f}пикс, отверстие={is_hole}, точек={len(points_m)}")

        logger.info(f"Отфильтровано контуров: {len(filtered_contours)}")
        return filtered_contours

    def contours_to_polygons(self, contours: List[Tuple[np.ndarray, bool]]) -> Union[Polygon, MultiPolygon]:
        """
        Конвертирует контуры в полигоны Shapely

        Args:
            contours: Список (контур, is_hole)

        Returns:
            Polygon или MultiPolygon
        """
        logger.info("Конвертация контуров в полигоны...")

        # Разделяем внешние контуры и отверстия
        exteriors = []
        holes = []

        for contour_points, is_hole in contours:
            if len(contour_points) < 3:
                continue

            coords = [(float(x), float(y)) for x, y in contour_points]

            # Убираем дубликаты подряд идущих точек
            unique_coords = [coords[0]]
            for coord in coords[1:]:
                if coord != unique_coords[-1]:
                    unique_coords.append(coord)

            if len(unique_coords) < 3:
                continue

            if is_hole:
                holes.append(unique_coords)
            else:
                exteriors.append(unique_coords)

        if not exteriors:
            raise ValueError("Не найдено внешних контуров")

        logger.info(f"Внешних контуров: {len(exteriors)}, отверстий: {len(holes)}")

        # Создаем полигоны
        polygons = []

        for ext_coords in exteriors:
            try:
                # Находим отверстия, которые находятся внутри этого внешнего контура
                exterior_poly = Polygon(ext_coords)
                if not exterior_poly.is_valid:
                    exterior_poly = exterior_poly.buffer(0)

                if exterior_poly.is_empty:
                    continue

                # Находим отверстия внутри этого полигона
                polygon_holes = []
                for hole_coords in holes:
                    try:
                        hole_poly = Polygon(hole_coords)
                        if hole_poly.is_valid and exterior_poly.contains(hole_poly):
                            polygon_holes.append(hole_coords)
                    except:
                        continue

                # Создаем полигон с отверстиями
                poly = Polygon(ext_coords, holes=polygon_holes if polygon_holes else None)

                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
                elif not poly.is_valid:
                    # Пытаемся исправить
                    fixed_poly = poly.buffer(0)
                    if fixed_poly.is_valid and not fixed_poly.is_empty:
                        polygons.append(fixed_poly)

            except Exception as e:
                logger.warning(f"Ошибка создания полигона: {e}")
                continue

        if not polygons:
            raise ValueError("Не удалось создать валидные полигоны")

        # Объединяем полигоны
        result = unary_union(polygons) if len(polygons) > 1 else polygons[0]

        logger.info(f"Создано полигонов: {len(polygons) if isinstance(result, MultiPolygon) else 1}")
        logger.info(f"Общая площадь: {result.area:.6f} м²")

        return result

    def process_image(self,
                      image_path: Union[str, Path],
                      target_size_m: float = 2.0,
                      min_area_px: int = 50,
                      morph_open: int = 0,
                      morph_close: int = 0,
                      invert: bool = False) -> Union[Polygon, MultiPolygon]:
        """
        Полный пайплайн обработки изображения

        Args:
            image_path: Путь к изображению
            target_size_m: Размер в метрах
            min_area_px: Минимальная площадь контура
            morph_open: Морфологическое открытие
            morph_close: Морфологическое закрытие
            invert: Инвертировать изображение

        Returns:
            Полигон или мультиполигон в метрах с центром в (0,0)
        """
        # Предобработка изображения
        binary = self.load_and_preprocess(
            image_path, target_size_m, morph_open, morph_close, invert
        )

        # Извлечение контуров
        contours = self.extract_contours(binary, min_area_px)

        # Конвертация в полигоны
        polygons = self.contours_to_polygons(contours)

        return polygons


# Пример использования
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    processor = ImageProcessor()

    # Пример обработки
    try:
        polygons = processor.process_image(
            "test_image.png",
            target_size_m=2.0,
            min_area_px=100,
            morph_open=3,
            morph_close=3,
            invert=True
        )
        print(f"Обработка завершена. Тип результата: {polygons.geom_type}")
        print(f"Площадь: {polygons.area:.6f} м²")
        print(f"Границы: {polygons.bounds}")

    except Exception as e:
        print(f"Ошибка: {e}")