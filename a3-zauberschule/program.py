import dataclasses
import heapq
import os
from typing import Iterable, List, Tuple

import numpy as np

# (Veränderung der Feld-Koordinaten, zu überprüfende Wand-Koordinate, Wegkosten)
STEPS = (
    (lambda i, j, k: ((i, j, k - 1), (i, 2 * j + 1, 2 * k)), 1),  # Schritt nach links
    (lambda i, j, k: ((i, j, k + 1), (i, 2 * j + 1, 2 * k + 2)), 1),  # Schritt nach rechts
    (lambda i, j, k: ((i, j - 1, k), (i, 2 * j, 2 * k + 1)), 1),  # Schritt nach oben
    (lambda i, j, k: ((i, j + 1, k), (i, 2 * j + 2, 2 * k + 1)), 1),  # Schritt nach unten
    (lambda i, j, k: ((i - 1, j, k), None), 3),  # Stockwerk-wechsel nach unten
    (lambda i, j, k: ((i + 1, j, k), None), 3),  # Stockwerk-wechsel nach oben
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
                    if c == "A":
                        start = (lv, int((ni - 1) / 2), int((mi - 1) / 2))
                    if c == "B":
                        end = (lv, int((ni - 1) / 2), int((mi - 1) / 2))

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


def main():
    n_bsp = input("Bitte Nummer des Beispiels eingeben:\n> ")
    n, m, room, start, end = load_zauberschule(f"input/zauberschule{n_bsp}.txt")

    # Die einzelnen Felder für den Dijkstra-Algorithmus
    # und das Zugehörige Array zum markieren besuchter Felder
    plan = np.empty((2, int((n - 1) / 2), int((m - 1) / 2)), "3uint16, uint64")
    seen = np.zeros((2, int((n - 1) / 2), int((m - 1) / 2)), bool)
    # heap als Schlange für den Algorithmus
    queue: List[DijkstraItem] = []
    # Erstes Item (Startpunkt) in die Schlange
    heapq.heappush(queue, DijkstraItem(0, start))
    seen[start] = True

    # run dijkstra and set the coord of the previous field at position of B
    n_steps = 0
    while len(queue) != 0:
        n_steps += 1
        item = heapq.heappop(queue)
        distance = item.distance
        coord = item.coord

        if coord == end:
            print(f"break found path with length {distance} in {n_steps} steps")
            break

        for fn, cost in STEPS:
            next_coord, to_check = fn(*coord)
            next_distance = distance + cost
            if (
                not in_bounds(next_coord, np.shape(plan))
                or (to_check and room[to_check] == "#")
                or not (not seen[next_coord] or plan[next_coord][1] > next_distance)
            ):
                continue
            # is this always the shortest path? #TODO no, because until B is choosen as the min item
            # in the heap queue, the field at next_coord could have been overwritten by a
            # shorter/longer path -> store the current cost in a field with plan, only overwrite if
            # lower cost
            # TODO idk
            plan[next_coord] = coord, next_distance
            seen[next_coord] = True

            if next_coord == end:
                print(f"would have found path in {n_steps}")
            heapq.heappush(queue, DijkstraItem(next_distance, next_coord))
    else:
        raise ValueError("Punkt B nicht gefunden!")

    # backtrace from B to A
    # print path


while True:
    try:
        main()
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
    print()
