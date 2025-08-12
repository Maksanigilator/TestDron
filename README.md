# Система дронов-художников

Модульная система для генерации траекторий рисования дронами на основе 2D изображений. Преобразует растровые изображения в последовательности команд MOVE/DRAW для автономных дронов-художников.

## Особенности

- 🖼️ **Обработка изображений**: Автоматическая обработка черно-белых изображений с морфологическими операциями
- 🎯 **Умная нарезка**: Генерация периметров и различных типов заполнения (zigzag, grid, concentric и др.)
- 🚁 **Управление дронами**: Полноценная модель дронов с характеристиками и ограничениями
- 📈 **Оптимизация**: Алгоритм ближайшего соседа для минимизации времени полета
- 📊 **Статистика**: Детальная аналитика по расходу краски, времени полета и батареи
- 🎨 **Многоцветность**: Поддержка нескольких дронов с разными цветами (в планах)

## Архитектура

Система состоит из 4 основных модулей:

### 1. `image_processor.py` - Обработчик изображений
- Загрузка и предобработка растровых изображений
- Морфологические операции (открытие/закрытие)
- Извлечение контуров и конвертация в полигоны Shapely
- Автоматическое масштабирование и центрирование

### 2. `slicing_engine.py` - Ядро нарезчика  
- Генерация периметров с заданной шириной линии
- Множество типов заполнения: lines, zigzag, grid, triangular, concentric, hilbert, octagram
- Вычисление областей заполнения с учетом периметров
- Использование pyclipper для точных геометрических операций

### 3. `drone.py` - Модель дрона
- Класс `ArtistDrone` с характеристиками и ограничениями
- Отслеживание расхода краски и заряда батареи
- Генерация команд MOVE/DRAW с координатами
- Статистика и визуализация траекторий

### 4. `trajectory_generator.py` - Генератор траекторий
- Объединение всех компонентов в единый пайплайн
- Оптимизация порядка сегментов (ближайший сосед)
- Поддержка нескольких дронов (в разработке)
- Сохранение результатов и статистики

## Установка

```bash
# Клонируем репозиторий
git clone <repository-url>
cd drone-artist-system

# Устанавливаем зависимости
pip install -r requirements.txt
```

### Зависимости

- `numpy` - численные вычисления
- `opencv-python` - обработка изображений  
- `shapely` - геометрические операции
- `pyclipper` - точные операции с полигонами
- `matplotlib` - визуализация (опционально)

## Быстрый старт

### 1. Подготовьте изображение
Создайте черно-белое изображение (PNG/JPG) с белым фоном и черным рисунком.

### 2. Настройте конфигурацию
Скопируйте `config.json` и настройте параметры:

```json
{
  "input_image": "path/to/your/image.png",
  "target_size_m": 2.0,
  "line_width": 0.02,
  "num_perimeters": 2,
  "infill_type": "zigzag",
  "drawing_height": 1.0,
  "output_dir": "./output"
}
```

### 3. Запустите генерацию
```bash
python trajectory_generator.py --config config.json --visualize
```

### 4. Получите результаты
- `trajectory_drone_01_black.txt` - файл с командами для дрона
- `trajectory_summary.json` - статистика проекта
- Визуализация траектории (при --visualize)

## Формат команд дрона

Система генерирует текстовые файлы с командами:

```
MOVE x y z        # Перемещение без рисования
DRAW x y z        # Рисование в точке
```

Пример:
```
MOVE 0.000000 0.000000 1.300000
MOVE 0.500000 0.500000 1.300000  
MOVE 0.500000 0.500000 1.000000
DRAW 0.500000 0.500000 1.000000
DRAW 1.500000 0.500000 1.000000
DRAW 1.500000 1.500000 1.000000
```

## Параметры конфигурации

### Обработка изображения
- `input_image` - путь к входному изображению
- `target_size_m` - размер наибольшей стороны в метрах
- `min_area_px` - минимальная площадь контура в пикселях
- `morph_open/close` - морфологические операции
- `invert_image` - инвертировать изображение

### Параметры нарезки
- `line_width` - ширина линии рисования (м)
- `num_perimeters` - количество периметров  
- `line_spacing` - расстояние между линиями заполнения (м)

### Заполнение
- `infill_type` - тип заполнения:
  - `lines` - прямые параллельные линии
  - `zigzag` - непрерывный зигзаг
  - `grid` - сетка (два направления)  
  - `triangular` - треугольная сетка
  - `concentric` - концентрические контуры
  - `hilbert` - кривая Гильберта (упрощенная)
  - `octagram` - восьмиугольная сетка
- `infill_angle` - угол поворота линий (градусы)
- `infill_density` - плотность заполнения (0.0-1.0)

### Параметры дрона
- `drawing_height` - высота рисования (м)
- `transit_height` - высота перемещения (м) 
- `drawing_speed` - скорость рисования (м/с)
- `drone_color` - цвет краски

## Примеры использования

### Базовое использование из кода

```python
from trajectory_generator import TrajectoryGenerator

config = {
    "input_image": "drawing.png",
    "target_size_m": 1.5,
    "line_width": 0.02,
    "num_perimeters": 3,
    "infill_type": "zigzag",
    "output_dir": "./output"
}

generator = TrajectoryGenerator(config)
drones = generator.generate_full_trajectory()

# Получаем статистику
for drone in drones:
    stats = drone.get_statistics()
    print(f"Дрон {drone.id}: {stats['total_distance_m']:.3f}м")
```

### Создание дрона напрямую

```python
from drone import ArtistDrone, DroneColor

drone = ArtistDrone(
    drone_id="test_drone",
    color=DroneColor.BLACK,
    start_position=(0, 0)
)

# Рисуем квадрат
square = [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)]
drone.add_trajectory_segment(square)
drone.return_to_start()

# Сохраняем и показываем статистику  
drone.save_trajectory("square.txt")
print(drone.get_statistics())
```

### Обработка изображения

```python
from image_processor import ImageProcessor

processor = ImageProcessor()
polygons = processor.process_image(
    image_path="drawing.png",
    target_size_m=2.0,
    morph_open=3,
    morph_close=3
)

print(f"Площадь: {polygons.area:.6f} м²")
```

### Настройка слайсера

```python
from slicing_engine import SlicingEngine, InfillType

engine = SlicingEngine()

# Периметры
perimeters = engine.generate_perimeters(
    polygons, 
    line_width=0.02, 
    num_perimeters=3
)

# Заполнение
infill_area = engine.compute_infill_area(polygons, 0.02, 3)
infill = engine.generate_infill(
    infill_area,
    InfillType.ZIGZAG,
    line_spacing=0.05,
    angle_deg=45.0
)
```

## Расширение функциональности

### Добавление новых типов заполнения

```python
# В slicing_engine.py
class InfillType(Enum):
    CUSTOM = "custom"

def _generate_custom_infill(self, area, spacing):
    # Ваша реализация
    pass
```

### Настройка характеристик дрона

```python
from drone import DroneCapabilities

custom_caps = DroneCapabilities(
    max_speed=3.0,
    drawing_speed=0.3,
    paint_capacity=150.0,
    battery_capacity=6000.0
)

drone = ArtistDrone(
    drone_id="heavy_drone",
    color=DroneColor.RED,
    capabilities=custom_caps
)
```

## Оптимизация производительности

### Рекомендации по параметрам

- **line_width**: 0.01-0.03м (1-3см) для стандартных маркеров
- **line_spacing**: 0.8-1.2 × line_width для плотного заполнения  
- **num_perimeters**: 1-3 для большинства случаев
- **target_size_m**: 1-3м для оптимального времени полета

### Оптимизация больших изображений

```python
config = {
    "min_area_px": 200,      # Увеличить для фильтрации мелких деталей
    "morph_open": 5,         # Убрать шум
    "line_spacing": 0.08,    # Увеличить для ускорения
    "infill_density": 0.6    # Снизить плотность
}
```


```python
import logging
logging.basicConfig(level=logging.INFO)

# Или детальное логирование
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Тестирование

```bash
# Запуск примеров
python example_usage.py

# Создание тестового изображения
python -c "from example_usage import create_test_image; create_test_image()"

# Тестирование отдельных модулей
python image_processor.py
python slicing_engine.py  
python drone.py
```

## Лицензия

MIT License - см. файл LICENSE

