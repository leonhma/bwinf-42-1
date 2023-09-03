import dataclasses
from enum import Enum
import heapq
import os
from typing import Iterable, List, Tuple

import numpy as np


@dataclasses.dataclass
class Char:
    _FIELD = "."
    _WALL = "#"
    _START = "A"
    _END = "B"
    BL = "⮲"
    BT = "⇧"
    BR = "⮳"
    LT = "⮵"
    LR = "⇨"
    LB = "⮷"
    TR = "⮱"
    TB = "⇩"
    TL = "⮰"
    RB = "⮶"
    RL = "⇦"
    RT = "⮴"
    VU = "⊙"
    VD = "⊗"


# (Veränderung der Feld-Koordinaten, zu überprüfende Wand-Koordinate, Wegkosten)
STEPS = (
    (lambda i, j, k: ((i, j, k - 1), (i, 2 * j + 1, 2 * k)), 1),  # Schritt nach links
    (lambda i, j, k: ((i, j, k + 1), (i, 2 * j + 1, 2 * k + 2)), 1),  # Schritt nach rechts
    (lambda i, j, k: ((i, j - 1, k), (i, 2 * j, 2 * k + 1)), 1),  # Schritt nach oben
    (lambda i, j, k: ((i, j + 1, k), (i, 2 * j + 2, 2 * k + 1)), 1),  # Schritt nach unten
    (lambda i, j, k: ((i - 1, j, k), None), 3),  # Stockwerk-wechsel nach unten
    (lambda i, j, k: ((i + 1, j, k), None), 3),  # Stockwerk-wechsel nach oben
)

STEP_CHARS = {(0, 0, -1): Char.RL, (0, 0, 1): Char.LR, (0, -1, 0): Char.BT, (0, 1, 0): Char.TB}
PRETTY_KERNELS = (
    lambda x: x[0, 1] == Char.TB and x[1, 2] == Char.LR and Char.TR,
    lambda x: x[0, 1] == Char.TB and x[2, 1] == Char.TB and Char.TB,
    lambda x: x[0, 1] == Char.TB and x[1, 0] == Char.RL and Char.TL,
    lambda x: x[1, 2] == Char.RL and x[0, 1] == Char.BT and Char.RT,
    lambda x: x[1, 2] == Char.RL and x[1, 0] == Char.RL and Char.RL,
    lambda x: x[1, 2] == Char.RL and x[2, 1] == Char.TB and Char.RB,
    lambda x: x[2, 1] == Char.BT and x[1, 0] == Char.RL and Char.BL,
    lambda x: x[2, 1] == Char.BT and x[0, 1] == Char.BT and Char.BT,
    lambda x: x[2, 1] == Char.BT and x[1, 2] == Char.LR and Char.BR,
    lambda x: x[1, 0] == Char.LR and x[0, 1] == Char.BT and Char.LT,
    lambda x: x[1, 0] == Char.LR and x[1, 2] == Char.LR and Char.LR,
    lambda x: x[1, 0] == Char.LR and x[2, 1] == Char.TB and Char.LB,
)


def load_zauberschule(
    path: str,
) -> Tuple[int, int, np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Öffne ein Beispiel und gebe den Raumplan, sowie die Start- und Endposition zurück.

    Parameters
    ----------
    path : str
        Der Dateipfad der Beispieldatei relativ zur `program.py`-Datei.

    Returns
    -------
    Tuple[int, int, np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]
        Ein Tuple bestehend aus:
          - int n
          - int m
          - Raumplan: 3-dimensionales Tuple, in dem Wände mit `False` und Gänge mit `True`
            markiert sind.
          - Der Startposition: Koordinate im Raumplan.
          - Der Endposition: Koordinate im Raumplan.
    """
    with open(os.path.join(os.path.dirname(__file__), path), "r", encoding="utf8") as f:
        n, m = map(int, f.readline().split())

        start, end = None, None
        room = np.empty((2, n, m), dtype=str)

        def load(lv: int):
            nonlocal start, end
            for ni in range(n):
                for mi, c in enumerate(f.readline()[:m]):
                    room[lv][ni][mi] = c
                    if c == Char._START:
                        start = (lv, ni, mi)
                    if c == Char._END:
                        end = (lv, ni, mi)

        load(1)
        f.readline()
        load(0)

        assert None not in (start, end), "Punkt A oder B konnten nicht gefunden werden!"

        return n, m, room, start, end


@dataclasses.dataclass(order=True)
class DijkstraItem:
    """
    Ein Eintrag in der Dijkstra-Queue.

    Attributes
    ----------
    distance: int
        Die Distanz von Punkt A zur Koordinate `coord`.
    coord: Tuple[int, int, int]
        Die Kordinate des Felds.
    """

    distance: int
    coord: Tuple[int, int, int] = dataclasses.field(compare=False)
    prev_coord: Tuple[int, int, int] = dataclasses.field(compare=False)


def in_bounds(coord: Iterable[int], max_: Iterable[int], min_: Iterable[int] = None) -> bool:
    """
    Überprüfe, ob eine Koordinate innerhalb der Grenzen eines n-dimensionalen Arrays liegt,
        die durch `min_` und `max_` angegeben werden.

    Parameters
    ----------
    coord : Iterable[int]
        Die Koordinate (eg. "(1, 2, 3)").
    max_ : Iterable[int]
        Die maximalen Werte für die einzelnen Dimensionen (exklusiv) (eg. "(4, 5, 5)").
    min_ : Iterable[int], optional
        Die minimalen Werte für die einzelnen Dimensionen (eg. "(0, 0, 0)").

    Returns
    -------
    bool
        Ein Wahrheitswert, der besagt, ob `coord` innerhalb der gegebenen Grenzen liegt.
    """
    if min_ is None:
        min_ = (0,) * len(max_)

    for i, n in enumerate(coord):
        if n < min_[i] or n >= max_[i]:
            return False
    return True


def to_plan_coord(x: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Rechne eine `room`-Kordinate in eine `plan`-Koordinate um.

    Parameters
    ----------
    x : Tuple[int, int, int]
        Die `room`-Koordinate.

    Returns
    -------
    Tuple[int, int, int]
        Die `plan`-Koordinate.
    """
    return x[0], int((x[1] - 1) / 2), int((x[2] - 1) / 2)


def to_room_coord(x: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Rechne eine `plan`-Kordinate in eine `room`-Koordinate um.

    Parameters
    ----------
    x : Tuple[int, int, int]
        Die `plan`-Koordinate.

    Returns
    -------
    Tuple[int, int, int]
        Die `room`-Koordinate.
    """
    return x[0], x[1] * 2 + 1, x[2] * 2 + 1


def main():
    n_bsp = input("Bitte Nummer des Beispiels eingeben:\n> ")
    n, m, room, start, end = load_zauberschule(f"input/zauberschule{n_bsp}.txt")
    # Werte für die kleinere Matrix umrechen
    n_p, m_p = map(lambda x: int((x - 1) / 2), (n, m))
    start_p, end_p = map(to_plan_coord, (start, end))

    # Die einzelnen Felder für den Dijkstra-Algorithmus
    # und das Zugehörige Array zum markieren besuchter Felder
    plan = np.empty((2, n_p, m_p), "3uint16")
    seen = np.zeros((2, n_p, m_p), bool)
    # heap als Schlange für den Algorithmus
    queue: List[DijkstraItem] = []
    # Erstes Item (Startpunkt) in die Schlange
    heapq.heappush(queue, DijkstraItem(0, start_p, (0, 0, 0)))

    # run dijkstra and set the coord of the previous field at position of B
    n_steps = 0
    while len(queue) != 0:
        n_steps += 1
        item = heapq.heappop(queue)
        distance = item.distance
        coord = item.coord
        if seen[coord]:
            continue
        seen[coord] = True
        plan[coord] = item.prev_coord

        if coord == end_p:
            print(f"break found path with length {distance} in {n_steps} steps")
            break

        for fn, cost in STEPS:
            next_coord, to_check = fn(*coord)
            next_distance = distance + cost
            if not in_bounds(next_coord, np.shape(plan)) or (
                to_check and room[to_check] == Char._WALL
            ):
                continue
            heapq.heappush(queue, DijkstraItem(next_distance, next_coord, coord))
    else:
        raise ValueError("Es wurde kein Pfad gefunden!")

    for lv in range(np.shape(plan)[0] - 1, -1, -1):
        print("-" * (np.shape(plan)[2] + 4))
        for n in range(np.shape(plan)[1]):
            print("| ", end="")
            for m in range(np.shape(plan)[2]):
                v = plan[lv, n, m]
                char = " "
                if not seen[lv, n, m]:
                    pass
                elif v[0] != lv:
                    char = "!"
                elif v[1] < n:
                    char = "⌄"
                elif v[1] > n:
                    char = "^"
                elif v[2] < m:
                    char = ">"
                elif v[2] > m:
                    char = "<"
                print(f"{char}", end="")
            print(" |")
        print("-" * (np.shape(plan)[2] + 4))

    # Verfolgen des Pfades und einfügen der Wegmarkierungen
    prev_p = end_p
    while True:
        current_p = tuple(plan[prev_p])
        current_r = to_room_coord(current_p)

        step = tuple(np.subtract(prev_p, current_p, dtype="i8"))
        if step[0] == 0:
            room[tuple(np.add(current_r, step))] = STEP_CHARS[step]
        else:
            if step[0] == 1:
                if room[0, current_r[1], current_r[2]] == Char._FIELD:
                    room[0, current_r[1], current_r[2]] = Char.VU
                if room[1, current_r[1], current_r[2]] == Char._FIELD:
                    room[1, current_r[1], current_r[2]] = Char.VU
            elif step[0] == -1:
                if room[0, current_r[1], current_r[2]] == Char._FIELD:
                    room[0, current_r[1], current_r[2]] = Char.VD
                if room[1, current_r[1], current_r[2]] == Char._FIELD:
                    room[1, current_r[1], current_r[2]] = Char.VD

        if current_p == start_p:
            break
        prev_p = current_p

    for lv in range(2):
        for n_ci in range(1, np.shape(room)[1], 2):
            for m_ci in range(1, np.shape(room)[2], 2):
                space = room[lv, (n_ci - 1) : (n_ci + 2), (m_ci - 1) : (m_ci + 2)]
                for kernel in PRETTY_KERNELS:
                    if char := kernel(space):
                        room[lv, n_ci, m_ci] = char
                        print(f'setting char "{char}" at position {(lv, n_ci, m_ci)}')
                        break

    for li in room[::-1]:
        for ni in li:
            for mi in ni:
                print(mi, end="")
            print()
        print()


while True:
    try:
        main()
        # arr = np.array([["#", "⇩", "#"], ["#", ".", "."], ["#", "⇩", "#"]])
        # for kernel in PRETTY_KERNELS:
        #     print(kernel(arr))
        # break
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
    print()
