#!/usr/bin/env python3
"""
trajectory_generator.py - Генератор траекторий для дронов-художников
Объединяет все компоненты для создания полной траектории рисования
"""
import json
import math
from typing import List, Tuple, Dict, Any, Union, Optional
from pathlib import Path
import logging

from image_processor import ImageProcessor
from slicing_engine import SlicingEngine, InfillType
from drone import ArtistDrone, DroneColor, DroneCapabilities

logger = logging.getLogger(__name__)


class TrajectoryOptimizer:
    """Оптимизатор траекторий для минимизации времени полета"""

    def __init__(self):
        pass

    def order_segments_nearest_neighbor(self,
                                        segments: List[List[Tuple[float, float]]],
                                        start_position: Tuple[float, float] = (0.0, 0.0)) -> List[
        List[Tuple[float, float]]]:
        """
        Упорядочивает сегменты по алгоритму ближайшего соседа

        Args:
            segments: Список сегментов траектории
            start_position: Стартовая позиция

        Returns:
            Упорядоченный список сегментов
        """
        if not segments:
            return []

        logger.info(f"Оптимизация порядка {len(segments)} сегментов")

        remaining = segments.copy()
        ordered = []
        current_pos = start_position

        while remaining:
            best_idx = 0
            best_distance = float('inf')
            best_reverse = False

            for i, segment in enumerate(remaining):
                if len(segment) < 2:
                    continue

                # Проверяем расстояние до начала и конца сегмента
                start_dist = self._distance_2d(current_pos, segment[0])
                end_dist = self._distance_2d(current_pos, segment[-1])

                if start_dist < best_distance:
                    best_distance = start_dist
                    best_idx = i
                    best_reverse = False

                if end_dist < best_distance:
                    best_distance = end_dist
                    best_idx = i
                    best_reverse = True

            # Берем лучший сегмент
            segment = remaining.pop(best_idx)

            if best_reverse:
                segment = list(reversed(segment))

            ordered.append(segment)
            current_pos = segment[-1]

        logger.info("Оптимизация порядка завершена")
        return ordered

    def _distance_2d(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Вычисляет 2D расстояние между точками"""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def optimize_for_multiple_drones(self,
                                     segments: List[List[Tuple[float, float]]],
                                     num_drones: int = 1) -> List[List[List[Tuple[float, float]]]]:
        """
        Распределяет сегменты между несколькими дронами для параллельного рисования

        Args:
            segments: Список всех сегментов
            num_drones: Количество дронов

        Returns:
            Список траекторий для каждого дрона
        """
        if num_drones <= 1:
            return [segments]

        logger.info(f"Распределение {len(segments)} сегментов между {num_drones} дронами")

        # Простой алгоритм: распределяем по принципу round-robin
        # В будущем можно добавить более сложную оптимизацию
        drone_segments = [[] for _ in range(num_drones)]

        for i, segment in enumerate(segments):
            drone_idx = i % num_drones
            drone_segments[drone_idx].append(segment)

        # Оптимизируем каждую траекторию отдельно
        optimized_trajectories = []
        for i, drone_segs in enumerate(drone_segments):
            if drone_segs:
                start_pos = (i * 0.5, 0.0)  # Разные стартовые позиции для дронов
                optimized = self.order_segments_nearest_neighbor(drone_segs, start_pos)
                optimized_trajectories.append(optimized)
            else:
                optimized_trajectories.append([])

        logger.info(f"Распределение завершено")
        return optimized_trajectories


class TrajectoryGenerator:
    """Главный класс генератора траекторий"""

    def __init__(self, config: Dict[str, Any]):
        """
        Инициализирует генератор с конфигурацией

        Args:
            config: Словарь с настройками
        """
        self.config = config
        self.image_processor = ImageProcessor()
        self.slicing_engine = SlicingEngine()
        self.optimizer = TrajectoryOptimizer()

        # Загружаем параметры из конфига
        self._load_config()

        logger.info("Генератор траекторий инициализирован")

    def _load_config(self):
        """Загружает параметры из конфигурации"""
        # Параметры изображения
        self.input_image = self.config['input_image']
        self.target_size_m = float(self.config.get('target_size_m', 2.0))
        self.min_area_px = int(self.config.get('min_area_px', 100))
        self.morph_open = int(self.config.get('morph_open', 0))
        self.morph_close = int(self.config.get('morph_close', 0))
        self.invert_image = bool(self.config.get('invert_image', False))

        # Параметры нарезки
        self.line_width = float(self.config.get('line_width', 0.02))
        self.num_perimeters = int(self.config.get('num_perimeters', 2))
        self.line_spacing = float(self.config.get('line_spacing', 0.05))

        # Параметры заполнения
        infill_type_str = self.config.get('infill_type', 'zigzag')
        self.infill_type = InfillType(infill_type_str)
        self.infill_angle = float(self.config.get('infill_angle', 0.0))
        self.infill_density = float(self.config.get('infill_density', 1.0))

        # Параметры дронов
        self.num_drones = int(self.config.get('num_drones', 1))
        self.drone_color = self.config.get('drone_color', 'black')
        self.drawing_height = float(self.config.get('drawing_height', 1.0))
        self.transit_height = float(self.config.get('transit_height', 1.3))
        self.drawing_speed = float(self.config.get('drawing_speed', 0.5))

        # Выходные файлы
        self.output_dir = Path(self.config.get('output_dir', './output'))
        self.output_prefix = self.config.get('output_prefix', 'trajectory')

        # Создаем выходную директорию
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_image_to_polygons(self):
        """Обрабатывает изображение в полигоны"""
        logger.info(f"Обработка изображения: {self.input_image}")

        try:
            polygons = self.image_processor.process_image(
                image_path=self.input_image,
                target_size_m=self.target_size_m,
                min_area_px=self.min_area_px,
                morph_open=self.morph_open,
                morph_close=self.morph_close,
                invert=self.invert_image
            )

            logger.info(f"Полигоны созданы: тип={polygons.geom_type}, площадь={polygons.area:.6f} м²")
            return polygons

        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {e}")
            raise

    def generate_toolpaths(self, polygons):
        """Генерирует траектории инструмента из полигонов"""
        logger.info("Генерация траекторий инструмента")

        all_segments = []

        try:
            # Генерируем периметры
            logger.info("Генерация периметров...")
            perimeters = self.slicing_engine.generate_perimeters(
                polygons,
                line_width=self.line_width,
                num_perimeters=self.num_perimeters,
                start_from_outside=True
            )

            logger.info(f"Сгенерировано периметров: {len(perimeters)}")
            all_segments.extend(perimeters)

            # Генерируем заполнение
            if self.infill_density > 0:
                logger.info("Генерация заполнения...")

                infill_area = self.slicing_engine.compute_infill_area(
                    polygons,
                    line_width=self.line_width,
                    num_perimeters=self.num_perimeters
                )

                if infill_area and not infill_area.is_empty:
                    infill_segments = self.slicing_engine.generate_infill(
                        infill_area,
                        infill_type=self.infill_type,
                        line_spacing=self.line_spacing,
                        angle_deg=self.infill_angle,
                        density=self.infill_density
                    )

                    logger.info(f"Сгенерировано линий заполнения: {len(infill_segments)}")
                    all_segments.extend(infill_segments)
                else:
                    logger.info("Область заполнения пуста - пропускаем заполнение")

            logger.info(f"Всего сегментов траектории: {len(all_segments)}")
            return all_segments

        except Exception as e:
            logger.error(f"Ошибка генерации траекторий: {e}")
            raise

    def create_drones(self) -> List[ArtistDrone]:
        """Создает дронов с заданными параметрами"""
        logger.info(f"Создание {self.num_drones} дронов")

        drones = []
        available_colors = [DroneColor.BLACK, DroneColor.RED, DroneColor.BLUE,
                            DroneColor.GREEN, DroneColor.YELLOW, DroneColor.PURPLE]

        # Создаем характеристики дронов
        capabilities = DroneCapabilities(
            drawing_speed=self.drawing_speed,
            min_drawing_height=self.drawing_height,
            transit_height=self.transit_height
        )

        for i in range(self.num_drones):
            # Определяем цвет
            if self.num_drones == 1:
                color = DroneColor(self.drone_color) if self.drone_color in [c.value for c in
                                                                             DroneColor] else DroneColor.BLACK
            else:
                color = available_colors[i % len(available_colors)]

            # Определяем стартовую позицию (разнесем дронов)
            start_x = i * 0.5
            start_y = 0.0

            # Определяем выходной файл
            output_file = self.output_dir / f"{self.output_prefix}_drone_{i + 1:02d}_{color.value}.txt"

            drone = ArtistDrone(
                drone_id=f"drone_{i + 1:02d}",
                color=color,
                start_position=(start_x, start_y),
                output_file=str(output_file),
                capabilities=capabilities
            )

            drones.append(drone)

        logger.info(f"Создано дронов: {len(drones)}")
        return drones

    def assign_trajectories_to_drones(self,
                                      segments: List[List[Tuple[float, float]]],
                                      drones: List[ArtistDrone]) -> None:
        """Назначает траектории дронам"""
        logger.info("Назначение траекторий дронам")

        if not drones:
            logger.warning("Нет дронов для назначения траекторий")
            return

        # Получаем стартовые позиции дронов
        start_positions = [(drone.start_x, drone.start_y) for drone in drones]

        # Распределяем сегменты между дронами и оптимизируем
        if len(drones) == 1:
            # Один дрон - оптимизируем всю траекторию
            optimized_segments = self.optimizer.order_segments_nearest_neighbor(
                segments, start_positions[0]
            )
            drone_trajectories = [optimized_segments]
        else:
            # Несколько дронов - распределяем и оптимизируем
            drone_trajectories = self.optimizer.optimize_for_multiple_drones(
                segments, len(drones)
            )

        # Назначаем траектории дронам
        for i, (drone, trajectory) in enumerate(zip(drones, drone_trajectories)):
            if trajectory:
                logger.info(f"Назначение {len(trajectory)} сегментов дрону {drone.id}")
                drone.add_trajectory_segments(
                    trajectory,
                    drawing_height=self.drawing_height,
                    drawing_speed=self.drawing_speed
                )
                drone.return_to_start()
            else:
                logger.warning(f"Дрону {drone.id} не назначено сегментов")

    def generate_full_trajectory(self) -> List[ArtistDrone]:
        """Выполняет полный цикл генерации траектории"""
        logger.info("Начало генерации полной траектории")

        try:
            # 1. Обрабатываем изображение
            polygons = self.process_image_to_polygons()

            # 2. Генерируем траектории инструмента
            segments = self.generate_toolpaths(polygons)

            if not segments:
                logger.warning("Не сгенерировано сегментов траектории")
                return []

            # 3. Создаем дронов
            drones = self.create_drones()

            # 4. Назначаем траектории дронам
            self.assign_trajectories_to_drones(segments, drones)

            # 5. Сохраняем траектории
            for drone in drones:
                drone.save_trajectory()

                # Выводим статистику
                stats = drone.get_statistics()
                logger.info(f"Статистика дрона {drone.id}:")
                for key, value in stats.items():
                    logger.info(f"  {key}: {value}")

            # 6. Сохраняем общую статистику
            self._save_summary_statistics(drones)

            logger.info("Генерация траектории завершена успешно")
            return drones

        except Exception as e:
            logger.error(f"Ошибка генерации траектории: {e}")
            raise

    def _save_summary_statistics(self, drones: List[ArtistDrone]) -> None:
        """Сохраняет общую статистику проекта"""
        summary_file = self.output_dir / f"{self.output_prefix}_summary.json"

        try:
            summary_data = {
                "project_info": {
                    "input_image": str(self.input_image),
                    "target_size_m": self.target_size_m,
                    "num_drones": len(drones),
                    "line_width": self.line_width,
                    "num_perimeters": self.num_perimeters,
                    "infill_type": self.infill_type.value,
                    "infill_density": self.infill_density
                },
                "drones": []
            }

            total_distance = 0.0
            total_drawing_distance = 0.0
            total_paint = 0.0

            for drone in drones:
                stats = drone.get_statistics()
                summary_data["drones"].append(stats)
                total_distance += stats["total_distance_m"]
                total_drawing_distance += stats["drawing_distance_m"]
                total_paint += stats["paint_used_ml"]

            summary_data["totals"] = {
                "total_distance_m": round(total_distance, 3),
                "total_drawing_distance_m": round(total_drawing_distance, 3),
                "total_paint_used_ml": round(total_paint, 2),
                "drawing_efficiency": round(total_drawing_distance / total_distance, 3) if total_distance > 0 else 0
            }

            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Сводная статистика сохранена: {summary_file}")

        except Exception as e:
            logger.warning(f"Не удалось сохранить сводную статистику: {e}")


def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурацию из файла"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Конфигурация загружена: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        raise


def main():
    """Главная функция для запуска из командной строки"""
    import argparse

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Парсинг аргументов
    parser = argparse.ArgumentParser(description="Генератор траекторий для дронов-художников")
    parser.add_argument('--config', '-c',
                        help="Путь к файлу конфигурации JSON",
                        default='config.json')
    parser.add_argument('--visualize', '-v',
                        help="Создать визуализацию траекторий",
                        action='store_true')

    args = parser.parse_args()

    try:
        # Загружаем конфигурацию
        config = load_config(args.config)

        # Создаем генератор
        generator = TrajectoryGenerator(config)

        # Генерируем траектории
        drones = generator.generate_full_trajectory()

        if not drones:
            logger.error("Не удалось создать траектории дронов")
            return 1

        # Визуализация (если запрошена)
        if args.visualize:
            logger.info("Создание визуализаций...")
            for drone in drones:
                try:
                    drone.visualize_trajectory(save_plot=True)
                except Exception as e:
                    logger.warning(f"Не удалось создать визуализацию для дрона {drone.id}: {e}")

        logger.info("Обработка завершена успешно!")
        return 0

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())