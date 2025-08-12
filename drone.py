#!/usr/bin/env python3
"""
drone.py - Класс дрона-художника
Представляет дрон с его параметрами и возможностями
"""
import math
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DroneColor(Enum):
    """Доступные цвета для дронов"""
    BLACK = "black"
    RED = "red"
    BLUE = "blue"
    GREEN = "green"
    YELLOW = "yellow"
    PURPLE = "purple"
    ORANGE = "orange"
    WHITE = "white"

    @property
    def rgb(self) -> Tuple[int, int, int]:
        """Возвращает RGB значения цвета"""
        color_map = {
            DroneColor.BLACK: (0, 0, 0),
            DroneColor.RED: (255, 0, 0),
            DroneColor.BLUE: (0, 0, 255),
            DroneColor.GREEN: (0, 255, 0),
            DroneColor.YELLOW: (255, 255, 0),
            DroneColor.PURPLE: (128, 0, 128),
            DroneColor.ORANGE: (255, 165, 0),
            DroneColor.WHITE: (255, 255, 255),
        }
        return color_map[self]


class DroneCommand:
    """Команда для дрона"""

    def __init__(self, command_type: str, x: float, y: float, z: float, speed: Optional[float] = None):
        self.type = command_type  # "MOVE" или "DRAW"
        self.x = x
        self.y = y
        self.z = z
        self.speed = speed

    def __str__(self) -> str:
        if self.speed is not None:
            return f"{self.type} {self.x:.6f} {self.y:.6f} {self.z:.6f} {self.speed:.3f}"
        return f"{self.type} {self.x:.6f} {self.y:.6f} {self.z:.6f}"

    def distance_to(self, other: 'DroneCommand') -> float:
        """Вычисляет 3D расстояние до другой команды"""
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )


class DroneCapabilities:
    """Характеристики и ограничения дрона"""

    def __init__(self,
                 max_speed: float = 2.0,  # м/с
                 max_acceleration: float = 1.0,  # м/с²
                 drawing_speed: float = 0.5,  # м/с при рисовании
                 min_drawing_height: float = 0.8,  # минимальная высота рисования
                 max_drawing_height: float = 1.2,  # максимальная высота рисования
                 transit_height: float = 1.5,  # высота перемещения
                 paint_capacity: float = 100.0,  # объем краски (мл)
                 paint_flow_rate: float = 0.1,  # расход краски мл/м
                 battery_capacity: float = 5000.0,  # емкость батареи (мАч)
                 power_consumption: float = 1000.0  # потребление при полете (мА)
                 ):
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.drawing_speed = drawing_speed
        self.min_drawing_height = min_drawing_height
        self.max_drawing_height = max_drawing_height
        self.transit_height = transit_height
        self.paint_capacity = paint_capacity
        self.paint_flow_rate = paint_flow_rate
        self.battery_capacity = battery_capacity
        self.power_consumption = power_consumption

    def estimate_paint_usage(self, drawing_distance: float) -> float:
        """Оценивает расход краски для заданной дистанции рисования"""
        return drawing_distance * self.paint_flow_rate

    def estimate_flight_time(self, total_distance: float, drawing_ratio: float = 0.5) -> float:
        """Оценивает время полета в минутах"""
        # Упрощенная оценка: учитываем среднюю скорость и потребление
        avg_speed = self.drawing_speed * drawing_ratio + self.max_speed * (1 - drawing_ratio)
        flight_time_hours = total_distance / (avg_speed * 3600)  # в часах
        return flight_time_hours * 60  # в минутах

    def estimate_battery_usage(self, flight_time_minutes: float) -> float:
        """Оценивает расход батареи в процентах"""
        consumption_per_minute = self.power_consumption / 60  # мА/мин
        total_consumption = consumption_per_minute * flight_time_minutes
        return (total_consumption / self.battery_capacity) * 100


class ArtistDrone:
    """Класс дрона-художника"""

    def __init__(self,
                 drone_id: str,
                 color: DroneColor,
                 start_position: Tuple[float, float] = (0.0, 0.0),
                 output_file: Optional[str] = None,
                 capabilities: Optional[DroneCapabilities] = None):
        """
        Инициализирует дрон-художника

        Args:
            drone_id: Уникальный идентификатор дрона
            color: Цвет краски дрона
            start_position: Начальная позиция на холсте (x, y)
            output_file: Путь к файлу для сохранения траектории
            capabilities: Характеристики дрона
        """
        self.id = drone_id
        self.color = color
        self.start_x, self.start_y = start_position
        self.output_file = output_file or f"trajectory_{drone_id}_{color.value}.txt"
        self.capabilities = capabilities or DroneCapabilities()

        # Текущее состояние
        self.current_x = start_position[0]
        self.current_y = start_position[1]
        self.current_z = self.capabilities.transit_height
        self.is_drawing = False

        # Статистика
        self.commands: List[DroneCommand] = []
        self.total_distance = 0.0
        self.drawing_distance = 0.0
        self.total_paint_used = 0.0

        logger.info(f"Создан дрон {self.id} с цветом {self.color.value} в позиции ({self.start_x}, {self.start_y})")

    def add_command(self, command: DroneCommand) -> None:
        """Добавляет команду в траекторию дрона"""
        if self.commands:
            # Вычисляем расстояние от предыдущей команды
            prev_cmd = self.commands[-1]
            distance = prev_cmd.distance_to(command)
            self.total_distance += distance

            if command.type == "DRAW":
                self.drawing_distance += distance
                self.total_paint_used += self.capabilities.estimate_paint_usage(distance)

        self.commands.append(command)

        # Обновляем текущую позицию
        self.current_x = command.x
        self.current_y = command.y
        self.current_z = command.z
        self.is_drawing = (command.type == "DRAW")

    def move_to(self, x: float, y: float, z: Optional[float] = None, speed: Optional[float] = None) -> None:
        """Добавляет команду перемещения"""
        if z is None:
            z = self.capabilities.transit_height

        command = DroneCommand("MOVE", x, y, z, speed)
        self.add_command(command)

    def draw_to(self, x: float, y: float, z: Optional[float] = None, speed: Optional[float] = None) -> None:
        """Добавляет команду рисования"""
        if z is None:
            z = self.capabilities.min_drawing_height

        if speed is None:
            speed = self.capabilities.drawing_speed

        command = DroneCommand("DRAW", x, y, z, speed)
        self.add_command(command)

    def add_trajectory_segment(self,
                               segment: List[Tuple[float, float]],
                               drawing_height: Optional[float] = None,
                               drawing_speed: Optional[float] = None) -> None:
        """
        Добавляет сегмент траектории (последовательность точек для рисования)

        Args:
            segment: Список координат (x, y)
            drawing_height: Высота рисования
            drawing_speed: Скорость рисования
        """
        if not segment or len(segment) < 2:
            return

        if drawing_height is None:
            drawing_height = self.capabilities.min_drawing_height

        if drawing_speed is None:
            drawing_speed = self.capabilities.drawing_speed

        # Перемещаемся к началу сегмента на высоте транзита
        start_x, start_y = segment[0]
        self.move_to(start_x, start_y, self.capabilities.transit_height)

        # Опускаемся для рисования
        self.move_to(start_x, start_y, drawing_height)

        # Рисуем сегмент
        for x, y in segment:
            self.draw_to(x, y, drawing_height, drawing_speed)

        # Поднимаемся после рисования
        end_x, end_y = segment[-1]
        self.move_to(end_x, end_y, self.capabilities.transit_height)

    def add_trajectory_segments(self,
                                segments: List[List[Tuple[float, float]]],
                                drawing_height: Optional[float] = None,
                                drawing_speed: Optional[float] = None) -> None:
        """Добавляет множество сегментов траектории"""
        for segment in segments:
            self.add_trajectory_segment(segment, drawing_height, drawing_speed)

    def return_to_start(self) -> None:
        """Возвращает дрон в стартовую позицию"""
        # Сначала поднимаемся
        self.move_to(self.current_x, self.current_y, self.capabilities.transit_height)
        # Затем летим в стартовую позицию
        self.move_to(self.start_x, self.start_y, self.capabilities.transit_height)
        # И опускаемся для посадки
        self.move_to(self.start_x, self.start_y, 0.2)

    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику выполнения траектории"""
        drawing_ratio = self.drawing_distance / self.total_distance if self.total_distance > 0 else 0
        flight_time = self.capabilities.estimate_flight_time(self.total_distance, drawing_ratio)
        battery_usage = self.capabilities.estimate_battery_usage(flight_time)

        return {
            "drone_id": self.id,
            "color": self.color.value,
            "total_commands": len(self.commands),
            "total_distance_m": round(self.total_distance, 3),
            "drawing_distance_m": round(self.drawing_distance, 3),
            "drawing_ratio": round(drawing_ratio, 3),
            "paint_used_ml": round(self.total_paint_used, 2),
            "paint_remaining_ml": round(self.capabilities.paint_capacity - self.total_paint_used, 2),
            "estimated_flight_time_min": round(flight_time, 1),
            "estimated_battery_usage_percent": round(battery_usage, 1),
            "can_complete": (
                    self.total_paint_used <= self.capabilities.paint_capacity and
                    battery_usage <= 100
            )
        }

    def save_trajectory(self, output_file: Optional[str] = None) -> None:
        """Сохраняет траекторию в файл"""
        filename = output_file or self.output_file

        logger.info(f"Сохранение траектории дрона {self.id} в {filename}")

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Заголовок файла
                f.write(f"# Траектория дрона {self.id}\n")
                f.write(f"# Цвет: {self.color.value}\n")
                f.write(f"# Команд: {len(self.commands)}\n")
                f.write(f"# Общая дистанция: {self.total_distance:.3f} м\n")
                f.write(f"# Дистанция рисования: {self.drawing_distance:.3f} м\n")
                f.write(f"# Расход краски: {self.total_paint_used:.2f} мл\n\n")

                # Команды
                for cmd in self.commands:
                    f.write(f"{cmd}\n")

            logger.info(f"Траектория сохранена: {len(self.commands)} команд")

        except Exception as e:
            logger.error(f"Ошибка сохранения траектории: {e}")
            raise

    def visualize_trajectory(self, save_plot: bool = False, plot_file: Optional[str] = None):
        """Визуализирует траекторию дрона (требует matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib не установлен, визуализация недоступна")
            return

        if not self.commands:
            logger.warning("Нет команд для визуализации")
            return

        # Разделяем команды на движения и рисование
        move_x, move_y = [], []
        draw_x, draw_y = [], []

        for cmd in self.commands:
            if cmd.type == "MOVE":
                move_x.append(cmd.x)
                move_y.append(cmd.y)
            else:
                draw_x.append(cmd.x)
                draw_y.append(cmd.y)

        # Создаем график
        fig, ax = plt.subplots(figsize=(10, 10))

        # Рисуем траекторию движения (серым)
        if move_x:
            ax.plot(move_x, move_y, 'gray', alpha=0.5, linewidth=1, label='Движение')

        # Рисуем траекторию рисования (цветом дрона)
        if draw_x:
            color_rgb = [c / 255.0 for c in self.color.rgb]
            ax.plot(draw_x, draw_y, color=color_rgb, linewidth=2, label='Рисование')

        # Отмечаем стартовую позицию
        ax.plot(self.start_x, self.start_y, 'go', markersize=8, label='Старт')

        ax.set_xlabel('X (метры)')
        ax.set_ylabel('Y (метры)')
        ax.set_title(f'Траектория дрона {self.id} ({self.color.value})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        if save_plot:
            plot_filename = plot_file or f"trajectory_{self.id}_{self.color.value}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            logger.info(f"График сохранен: {plot_filename}")

        plt.show()


# Пример использования
if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Создаем дрон
    drone = ArtistDrone(
        drone_id="drone_001",
        color=DroneColor.BLACK,
        start_position=(0.0, 0.0),
        output_file="test_trajectory.txt"
    )

    # Добавляем тестовую траекторию - квадрат
    square_points = [
        (0.5, 0.5),
        (1.5, 0.5),
        (1.5, 1.5),
        (0.5, 1.5),
        (0.5, 0.5)
    ]

    drone.add_trajectory_segment(square_points)
    drone.return_to_start()

    # Выводим статистику
    stats = drone.get_statistics()
    print("\nСтатистика дрона:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Сохраняем траекторию
    drone.save_trajectory()