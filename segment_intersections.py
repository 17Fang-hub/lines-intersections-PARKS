# -*- coding: utf-8 -*-
"""Підрахунок пар відрізків, що перетинаються: послідовно та через PARCS.

Локальний послідовний запуск без сторонніх бібліотек:
    python segment_intersections.py --segments-count 1000 --seed 42
    python segment_intersections.py --input input.json --output result.json

Для PARCS завантажте цей файл як Solution File, а JSON нижче як Input File:
    {"mode": "parallel", "segments_count": 10000,
     "seed": 42, "coordinate_limit": 100000}
Для послідовного запуску на master змініть mode на "sequential".
Штатний планувальник Hummer12007/parcs-python потребує підключеного воркера
навіть для sequential; сам послідовний алгоритм воркерів не викликає.

Кожна пара i < j перевіряється один раз. Дотик кінцями, накладання та
перетин у точці враховуються. Результат — кількість ПАР, а не унікальних
точок перетину. Геометрія використовує лише цілочисельні операції.

Master генерує весь набір до запуску алгоритму; за однакового seed та
версії Python обидва режими отримують однакові відрізки. Генерація вимірюється
окремо. elapsed_seconds включає передавання даних воркерам, обчислення та
збирання відповідей, але не генерацію, читання/запис файлів і запуск PARCS.

Синтаксис сумісний з Python 2.7 і Python 3. Для паралельного режиму потрібне
середовище PARCS із Pyro4 та асинхронними проксі, переданими планувальником.
"""

from __future__ import division, print_function

import argparse
import json
import random
import sys
import time

try:
    from Pyro4 import expose
    HAS_PYRO4 = True
except ImportError:
    HAS_PYRO4 = False

    def expose(method):
        return method

try:
    integer_types = (int, long)  # noqa: F821 - Python 2.7.
    iterate = xrange  # noqa: F821 - Python 2.7.
except NameError:
    integer_types = (int,)
    iterate = range

clock = getattr(time, "perf_counter", time.time)
CLOCK_NAME = "perf_counter" if hasattr(time, "perf_counter") else "time.time"


def validate_config(raw):
    """Перевіряємо JSON до генерації даних і вимірювання алгоритму."""
    if not isinstance(raw, dict):
        raise ValueError("Input must be a JSON object.")
    allowed = ("mode", "segments_count", "seed", "coordinate_limit")
    unknown = set(raw) - set(allowed)
    if unknown:
        raise ValueError("Unknown parameters: " + ", ".join(sorted(unknown)))

    config = {
        "mode": raw.get("mode", "sequential"),
        "segments_count": raw.get("segments_count"),
        "seed": raw.get("seed", 42),
        "coordinate_limit": raw.get("coordinate_limit", 100000),
    }
    if config["mode"] not in ("sequential", "parallel"):
        raise ValueError("mode must be 'sequential' or 'parallel'.")
    for key in ("segments_count", "seed", "coordinate_limit"):
        if isinstance(config[key], bool) or not isinstance(config[key], integer_types):
            raise ValueError(key + " must be an integer.")
    if config["segments_count"] < 2:
        raise ValueError("segments_count must be at least 2.")
    if config["coordinate_limit"] < 1:
        raise ValueError("coordinate_limit must be positive.")
    return config


def generate_segments(config):
    """Відрізок — (x1, y1, x2, y2); координати від 0 до limit включно."""
    generator = random.Random(config["seed"])
    limit = config["coordinate_limit"]
    segments = []
    for _ in iterate(config["segments_count"]):
        x1, y1 = generator.randint(0, limit), generator.randint(0, limit)
        x2, y2 = generator.randint(0, limit), generator.randint(0, limit)
        # Не генеруємо відрізки нульової довжини.
        while x1 == x2 and y1 == y2:
            x2, y2 = generator.randint(0, limit), generator.randint(0, limit)
        segments.append((x1, y1, x2, y2))
    return segments


def segments_intersect(first, second):
    """Точна перевірка перетину ненульових відрізків із цілими координатами."""
    x1, y1, x2, y2 = first
    x3, y3, x4, y4 = second

    # Межі спільної частини прямокутників, що охоплюють відрізки.
    left = max(min(x1, x2), min(x3, x4))
    right = min(max(x1, x2), max(x3, x4))
    bottom = max(min(y1, y2), min(y3, y4))
    top = min(max(y1, y2), max(y3, y4))
    if left > right or bottom > top:
        return False

    # Рівняння прямих: a*x + b*y + c = 0.
    a1, b1, c1 = y1 - y2, x2 - x1, x1 * y2 - x2 * y1
    a2, b2, c2 = y3 - y4, x4 - x3, x3 * y4 - x4 * y3
    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        # Прямі збігаються, якщо початок другого відрізка лежить на першій.
        # Перекриття їхніх діапазонів уже перевірено на початку.
        return a1 * x3 + b1 * y3 + c1 == 0

    # Формули Крамера: x = x_numerator / D, y = y_numerator / D.
    x_numerator = b1 * c2 - b2 * c1
    y_numerator = c1 * a2 - c2 * a1
    if determinant < 0:
        determinant = -determinant
        x_numerator, y_numerator = -x_numerator, -y_numerator

    # Множимо межі на додатний D замість ділення: жодної похибки float.
    return (left * determinant <= x_numerator <= right * determinant and
            bottom * determinant <= y_numerator <= top * determinant)


def pair_from_index(n, pair_index):
    """Номер пари -> (i, j) у порядку (0,1), (0,2), ..., (1,2), ... ."""
    total = n * (n - 1) // 2
    if not 0 <= pair_index < total:
        raise ValueError("Pair index is out of range.")

    # Перед рядком i розташовано i*(2*n-i-1)//2 пар.
    # Двійковий пошук потрібен лише для першої пари діапазону воркера.
    low, high = 0, n - 1
    while low + 1 < high:
        middle = (low + high) // 2
        before = middle * (2 * n - middle - 1) // 2
        if before <= pair_index:
            low = middle
        else:
            high = middle
    before = low * (2 * n - low - 1) // 2
    return low, low + 1 + pair_index - before


def count_pair_range(segments, first_pair, last_pair):
    """Спільне ядро двох режимів; last_pair не включається."""
    n = len(segments)
    total = n * (n - 1) // 2
    if not 0 <= first_pair <= last_pair <= total:
        raise ValueError("Invalid pair range.")
    if first_pair == last_pair:
        return 0, 0

    i, j = pair_from_index(n, first_pair)
    intersections = 0
    for _ in iterate(first_pair, last_pair):
        if segments_intersect(segments[i], segments[j]):
            intersections += 1
        j += 1
        if j == n:
            i += 1
            j = i + 1
    return intersections, last_pair - first_pair


def run_sequential(segments):
    """Перевіряємо всі пари в поточному процесі."""
    total = len(segments) * (len(segments) - 1) // 2
    return count_pair_range(segments, 0, total)


def run_parallel(segments, workers):
    """Розподіляємо пари порівну; кожен worker отримує всі відрізки."""
    if not HAS_PYRO4:
        raise RuntimeError("Parallel mode requires Pyro4 in the PARCS environment.")
    if not workers:
        raise RuntimeError("Parallel mode requires connected PARCS workers.")

    total = len(segments) * (len(segments) - 1) // 2
    workers_used = min(len(workers), total)
    pending_results = []
    for index in iterate(workers_used):
        first = total * index // workers_used
        last = total * (index + 1) // workers_used
        pending_results.append(workers[index].count_intersections(
            segments, first, last
        ))

    # Усі запити вже відправлені; тепер очікуємо їхні результати.
    intersections = 0
    checked = 0
    for pending in pending_results:
        partial_intersections, partial_checked = pending.value
        intersections += partial_intersections
        checked += partial_checked
    return intersections, checked, workers_used


def count_intersections(raw_config, workers=None):
    """Готуємо однакові дані та окремо вимірюємо обчислення обраного режиму."""
    config = validate_config(raw_config)
    generation_started = clock()
    segments = generate_segments(config)
    generation_elapsed = clock() - generation_started

    started = clock()
    if config["mode"] == "sequential":
        intersections, checked = run_sequential(segments)
        workers_used = 0
    else:
        intersections, checked, workers_used = run_parallel(segments, workers)
    elapsed = clock() - started

    total = len(segments) * (len(segments) - 1) // 2
    if checked != total or not 0 <= intersections <= checked:
        raise RuntimeError("Invalid result: pair counts are inconsistent.")
    return {
        "mode": config["mode"],
        "segments_count": len(segments),
        "total_pairs": total,
        "checked_pairs": checked,
        "intersecting_pairs": intersections,
        "elapsed_seconds": elapsed,
        "data_generation_seconds": generation_elapsed,
        "workers_used": workers_used,
        "seed": config["seed"],
        "coordinate_limit": config["coordinate_limit"],
        "timer": CLOCK_NAME,
    }


class Solver(object):
    """Контракт завантажуваної задачі Hummer12007/parcs-python."""

    def __init__(self, workers=None, input_file_name=None, output_file_name=None):
        self.workers = workers
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name

    def solve(self):
        with open(self.input_file_name, "r") as source:
            config = json.load(source)
        result = count_intersections(config, self.workers)
        text = json.dumps(result, indent=2, sort_keys=True)
        with open(self.output_file_name, "w") as destination:
            destination.write(text + "\n")
        print(text)
        return result

    @staticmethod
    @expose
    def count_intersections(segments, first_pair, last_pair):
        """Віддалений метод, що виконується на worker."""
        return count_pair_range(segments, first_pair, last_pair)


def main():
    parser = argparse.ArgumentParser(description="Segment intersections / PARCS-Python")
    parser.add_argument("--input", help="Input JSON; otherwise use CLI parameters.")
    parser.add_argument("--output", help="Save result JSON to this path.")
    parser.add_argument("--segments-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coordinate-limit", type=int, default=100000)
    args = parser.parse_args()

    try:
        if args.input:
            with open(args.input, "r") as source:
                config = json.load(source)
        else:
            config = {"mode": "sequential", "segments_count": args.segments_count,
                      "seed": args.seed, "coordinate_limit": args.coordinate_limit}
        if isinstance(config, dict) and config.get("mode") == "parallel":
            raise ValueError("Use the PARCS web interface for parallel mode; "
                             "this CLI runs the sequential version only.")
        result = count_intersections(config)
        text = json.dumps(result, indent=2, sort_keys=True)
        if args.output:
            with open(args.output, "w") as destination:
                destination.write(text + "\n")
        print(text)
        return 0
    except (ValueError, RuntimeError, IOError) as error:
        print("Error: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
