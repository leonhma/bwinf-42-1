import os
from typing import Dict, List, Tuple
from time import time


def load_tour(
    path: str,
) -> List[Tuple[str, int, bool, int]]:
    """
    Öffne ein Beispiel und gebe die Tour zurück.

    Parameters
    ----------
    path : str
        Der Dateipfad der Beispieldatei relativ zur `program.py`-Datei.

    Returns
    -------
    List[Tuple[str, int, bool, int]]
        Chronologisch sortierte Liste mit den Punkten der Tour.
        Ein Punkt ist ein Tuple mit:
        - dem Name des Ortes
        - dem Jahr der Besichtigung
        - ob der Ort essentiell ist
        - dem Abstand zum vorherigen Punkt
    """
    with open(os.path.join(os.path.dirname(__file__), path), "r", encoding="utf8") as f:
        # Dimensionen einlesen
        n = int(f.readline())

        tour = []
        for i in range(n):
            # Lesen einer Zeile
            name, year, ess, dist = f.readline().split(",")
            # Konvertieren zu Datentypen
            year = int(year)
            ess = ess == "X"
            dist = int(dist)
            # Hinzufügen zu Tour
            tour.append([name, [year], ess, dist])

        if not tour == (tour_ := list(sorted(tour, key=lambda x: x[1][0]))):
            print(
                "Die Eingabe-Tour war nicht chronologisch sortiert. "
                "Das könnte zu Problemen bei der kumulativen Distanz führen. "
                "Anyways, ..."
            )
            tour = tour_
    return tour


def main(tour: List[Tuple[str, List[int], bool, int]]):
    timed = time()  # Zeitmessung

    offset_dist = 0  # Variable, um entfernte Strecke zu speichern

    # Unwichtige Knoten vom Anfang entfernen
    x = 0
    while not tour[x][2]:
        x += 1
    offset_dist -= tour[x][3]
    tour = tour[x:]

    # Subtouren finden
    subtours: Dict[int, Tuple[int, int]] = {}  # {j: (i, v)}
    for i in range(len(tour) - 1):
        for j in range(i + 1, len(tour)):
            if tour[i][0] == tour[j][0]:
                subtours[j] = (i, tour[j][3] - tour[i][3])
                break
            elif tour[j][2]:
                break

    # Beste Subtouren-Kombination finden (Weighted Interval Scheduling)
    best = {-1: (0, [])}  # 1: (value, ((i, j), (i, j)))
    for j in range(len(tour)):
        if j in subtours:
            i, v = subtours[j]
            best[j] = max(
                best[j - 1], (best[i][0] + v, best[i][1] + [(i, j)]), key=lambda x: x[0]
            )
        else:
            best[j] = best[j - 1]  # nur leerraum

    # Subtouren entfernen
    for i, j in reversed(best[len(tour) - 1][1]):
        tour[j][1] = tour[i][1] + tour[j][1]
        offset_dist += tour[i][3] - tour[j][3]
        tour = tour[:i] + tour[j:]

    print(f"Berechnet in {(time() - timed)*1000:.2f}ms\n")
    # Tour ausgeben
    print(f"Tour:   (Gesamtlänge {tour[-1][3] + offset_dist})")
    for i, (name, years, *_) in enumerate(tour):
        print(f"{i+1}. {name} ({','.join(map(str, years))})")


# Haupt-Loop
while True:
    try:
        n_bsp = int(input("Bitte Nummer des Beispiels eingeben:\n> "))
        tour = load_tour(f"input/tour{n_bsp}.txt")
        print()
        main(tour)
    except TimeoutError as e:  # Error-Handling
        print(f"{e.__class__.__name__}: {e}")
    print()
