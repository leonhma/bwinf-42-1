import dataclasses
import heapq
from typing import Iterable, List, Tuple
import os
import numpy as np

STEPS = (  # (Veränderung der Koordinaten, Weg-Kosten)
    ((0, 0, -2), 1),
    ((0, 0, 2), 1),
    ((0, -2, 0), 1),
    ((0, 2, 0), 1),
    ((-1, 0, 0), 3),
    ((1, 0, 0), 3),
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
        room = np.empty((2, n, m), dtype=np.str_)

        def load(lv: int):
            nonlocal start, end
            for ni in range(n):
                for mi, c in enumerate(f.readline()[:m]):
                    room[lv][ni][mi] = c
                    if c == "A":
                        start = (lv, ni, mi)
                    if c == "B":
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
    prev_coords: List[Tuple[int, int, int]]
        Eine Liste der vorherigen besuchten Punkte in Reihenfolge. Das Feld `coord` nicht
        eingeschlossen.
    """

    distance: int
    coord: Tuple[int, int, int] = dataclasses.field(compare=False)


def in_bounds(coord: Iterable[int], min_: Iterable[int], max_: Iterable[int]) -> bool:
    """
    Überprüfe, ob eine Koordinate innerhalb der Grenzen eines n-dimensionalen Arrays liegt,
        die durch `min_` und `max_` angegeben werden.

    Parameters
    ----------
    coord : Iterable[int]
        Die Koordinate (eg. "(1, 2, 3)").
    min_ : Iterable[int]
        Die minimalen Werte für die einzelnen Dimensionen (eg. "(0, 0, 0)").
    max_ : Iterable[int]
        Die maximalen Werte für die einzelnen Dimensionen (exklusiv) (eg. "(4, 5, 5)").

    Returns
    -------
    bool
        Ein Wahrheitswert, der besagt, ob `coord` innerhalb der gegebenen Grenzen liegt.
    """
    for i, n in enumerate(coord):
        if n < min_[i] or n >= max_[i]:
            return False
    return True


def main():
    n_bsp = input("Bitte Nummer des Beispiels eingeben:\n> ")
    n, m, room, start, end = load_zauberschule(f"input/zauberschule{n_bsp}.txt")
    print(room)

    plan = np.zeros((2, n, m), "3uint16")
    seen = np.zeros((2, n, m), bool)
    seen[start] = True
    queue: List[DijkstraItem] = []
    heapq.heappush(queue, DijkstraItem(0, start))

    # run dijkstra and set the coord of the previous field at position of B
    n_steps = 0
    while len(queue) != 0:
        n_steps += 1
        item = heapq.heappop(queue)
        distance = item.distance
        coord = item.coord
        if coord == end:
            print(f'break found path with length {distance} in {n_steps} steps')
            break

        for step, cost in STEPS:
            next_coord = tuple(np.add(coord, step))
            if (
                not in_bounds(next_coord, (0, 0, 0), (2, n, m))
                or (step[0] == 0 and room[tuple(np.add(coord, np.floor_divide(step, 2)))] == '#')
                or seen[next_coord]
            ):
                continue
            plan[next_coord] = coord
            seen[next_coord] = True
            if next_coord == end:
                print(f'would have found path in {n_steps}')
            heapq.heappush(queue, DijkstraItem(distance + cost, next_coord))
    else:
        raise ValueError("Punkt B nicht gefunden!")

    # backtrace from B to A
    # print path


while True:
    try:
        main()
    except Exception as e:
        print(f'{e.__class__.__name__}: {e}')
    print()
