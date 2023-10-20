import dataclasses
import heapq
import os
import time
from typing import Iterable, List, Tuple

import numpy as np


@dataclasses.dataclass
class Char:
    _FIELD = "."
    _WALL = "#"
    _START = "A"
    _END = "B"
    BL = "<"
    BT = "^"
    BR = ">"
    LT = "^"
    LR = ">"
    LB = "v"
    TR = ">"
    TB = "v"
    TL = "<"
    RB = "v"
    RL = "<"
    RT = "^"
    VU = "!"
    VD = "!"


# (Veränderung der Feld-Koordinaten (plan, seen), zu überprüfende Wand-Koordinate (room), Wegkosten)
STEPS = (
    (lambda i, j, k: ((i, j, k - 1), (i, 2 * j + 1, 2 * k)), 1),  # Schritt nach links
    (lambda i, j, k: ((i, j, k + 1), (i, 2 * j + 1, 2 * k + 2)), 1),  # Schritt nach rechts
    (lambda i, j, k: ((i, j - 1, k), (i, 2 * j, 2 * k + 1)), 1),  # Schritt nach oben
    (lambda i, j, k: ((i, j + 1, k), (i, 2 * j + 2, 2 * k + 1)), 1),  # Schritt nach unten
    (lambda i, j, k: ((i - 1, j, k), None), 3),  # Stockwerk-wechsel nach unten
    (lambda i, j, k: ((i + 1, j, k), None), 3),  # Stockwerk-wechsel nach oben
)

# Helfer-Funktionen und Konstanten für die Ausgabe
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
) -> Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]:
    """
    Öffne ein Beispiel und gebe den Raumplan, sowie die Start- und Endposition zurück.

    Parameters
    ----------
    path : str
        Der Dateipfad der Beispieldatei relativ zur `program.py`-Datei.

    Returns
    -------
    Tuple[np.ndarray, Tuple[int, int, int], Tuple[int, int, int]]
        Ein Tuple bestehend aus:
          - Raumplan: 3-dimensionales numpy.ndarray, das die einzelnen Charaktere enthält.
          - Der Startposition: Koordinate im Raumplan.
          - Der Endposition: Koordinate im Raumplan.
    """
    with open(os.path.join(os.path.dirname(__file__), path), "r", encoding="utf8") as f:
        # Dimensionen einer Ebene einlesen
        n, m = map(int, f.readline().split())

        # Variablen für Start- und Endkoordinaten und den Raumplan
        start, end = None, None
        room = np.empty((2, n, m), dtype=str)

        def load(lv: int):
            nonlocal start, end
            for ni in range(n):
                for mi, c in enumerate(f.readline()[:m]):
                    room[lv][ni][mi] = c  # Charakter in `room` speichern
                    if c == Char._START:
                        start = (lv, ni, mi)  # Startposition speichern
                    if c == Char._END:
                        end = (lv, ni, mi)  # Endposition speichern

        load(1)  # Oberes Stockwerk laden
        f.readline()  # eine Leerzeile "verbrauchen"
        load(0)  # Unteres Stockwerk einlesen

        assert None not in (
            start,
            end,
        ), "Ungültiges Beispiel! (Punkt A oder B konnten nicht gefunden werden)"

        return room, start, end


@dataclasses.dataclass(order=True)
class DijkstraItem:
    """
    Ein Eintrag in der Dijkstra-Queue.

    Attributes
    ----------
    distance : int
        Die Distanz von Punkt A zur Koordinate `coord`.
    coord : Tuple[int, int, int]
        Die Koordinate des Felds.
    prev_coord : Tuple[int, int, int]
        Die Koordinate des Feldes bei dessen Besichtigung dieses Item
        in die queue hinzugefügt wurde. D. h. Diese Koordinate ist das nächste
        Feld auf dem kürzesten Weg in Richtung Startpunkt.
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


def dijkstra(
    room: np.ndarray,
    plan: np.ndarray,
    n_p: int,
    m_p: int,
    start_p: Tuple[int, int, int],
    end_p: Tuple[int, int, int],
):
    # Die einzelnen Felder für den Dijkstra-Algorithmus
    # und das Zugehörige Array zum markieren besuchter Felder
    seen = np.zeros((2, n_p, m_p), bool)
    # heap als Schlange für den Algorithmus
    queue: List[DijkstraItem] = []
    # Erstes Item (Startpunkt) in die Schlange
    heapq.heappush(queue, DijkstraItem(0, start_p, (0, 0, 0)))

    # Dijkstra-Algorithmus
    n_steps = 0
    while len(queue) != 0:
        n_steps += 1
        item = heapq.heappop(queue)
        distance = item.distance
        coord = item.coord
        # Wenn dieses Feld schon besucht wurde, weiter iterieren
        if seen[coord]:
            continue
        # Aktuelles Feld als besucht markieren
        seen[coord] = True
        # Vorgänger-Feld festhalten
        plan[coord] = item.prev_coord

        # Wenn der Endpunkt gefunden wurde, ist der Algorithmus fertig
        if coord == end_p:
            return n_steps, distance

        # Iterieren der verschiedenen Schritt-Möglichkeiten
        for fn, cost in STEPS:
            next_coord, to_check = fn(*coord)
            next_distance = distance + cost
            # wenn der Schritt außerhalb des Raums liegt oder eine Wand im Weg steht, überpringen
            if not in_bounds(next_coord, np.shape(plan)) or (
                to_check and room[to_check] == Char._WALL
            ):
                continue
            # mögliches Feld in die Schlange hinzufügen
            heapq.heappush(queue, DijkstraItem(next_distance, next_coord, coord))
    else:
        raise ValueError("Es wurde kein Pfad gefunden!")


# Zurückverfolgen des Pfades und einfügen der Wegmarkierungen
def trace(room, start_p, end_p, plan):
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


# einfügen von Markierungen zwischen den Feldern
def pretty(room):
    for lv in range(2):
        for n_ci in range(1, np.shape(room)[1], 2):
            for m_ci in range(1, np.shape(room)[2], 2):
                space = room[lv, (n_ci - 1) : (n_ci + 2), (m_ci - 1) : (m_ci + 2)]
                for kernel in PRETTY_KERNELS:
                    if char := kernel(space):
                        room[lv, n_ci, m_ci] = char
                        break


# Speichern des Pfades in eine Datei
def export(room: np.ndarray, path: str) -> str:
    p = os.path.join(os.path.dirname(__file__), path)
    with open(p, "w", encoding="utf8") as f:
        f.write(" ".join(map(str, np.shape(room)[1:])))
        f.write("\n")
        for li in room[::-1]:
            for ni in li:
                for mi in ni:
                    f.write(mi)
                f.write("\n")
            f.write("\n")
    return p


# Haupt-Loop
def main(room, start, end, n_bsp):
    t_s = time.time()  # Zeitmessung start

    # Werte für die kleinere Matrix umrechen
    n_p, m_p = map(lambda x: int((x - 1) / 2), np.shape(room)[1:])
    start_p, end_p = map(to_plan_coord, (start, end))
    plan = np.empty((2, n_p, m_p), "3uint16")
    n_steps, distance = dijkstra(room, plan, n_p, m_p, start_p, end_p)

    t_e = time.time()  # Zeitmessung ende

    # Verfolgen des berechneten Pfades und einfügen der Wegmarkierungen
    trace(room, start_p, end_p, plan)

    pretty(room)

    # Raum als ASCII-Art ausgeben
    if n_bsp <= 3:
        shape = np.shape(room)
        print(f'Oben:{" "*(shape[2]-4)}Unten:')
        for ni in range(shape[1]):
            for li in range(shape[0] - 1, -1, -1):
                for mi in range(shape[2]):
                    print(room[li, ni, mi], end="")
                print(" ", end="")
            print()

        print()

    # Exportieren nach Datei
    export(room, f"output/zauberschule{n_bsp}.txt")
    print(f'Ausgabe gespeichert in "output/zauberschule{n_bsp}.txt"')
    print()

    # Lösungswerte ausgeben
    print(
        f"Weg mit Länge {distance}s in {n_steps} Iterationen ({((t_e-t_s)*1000):.2f}ms) gefunden."
    )


while True:
    try:
        n_bsp = int(input("Bitte Nummer des Beispiels eingeben:\n> "))
        room, start, end = load_zauberschule(f"input/zauberschule{n_bsp}.txt")
        print()
        main(room, start, end, n_bsp)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
    print()
