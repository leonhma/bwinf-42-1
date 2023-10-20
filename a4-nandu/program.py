from itertools import product
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def load_board(
    path: str,
) -> Tuple[int, int, np.ndarray, List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Öffne ein Beispiel und gebe die Konstruktion zurück.

    Parameters
    ----------
    path : str
        Der Dateipfad der Beispieldatei relativ zur `program.py`-Datei.

    Returns
    -------
    Tuple[int, int, np.ndarray, List[Tuple[str, int]], List[Tuple[str, int]]]
        - n, m (Größen der Konstruktion)
        - Ein 2D-Array mit den Zeichen der Konstruktion.
        - Liste der Input-Positionen (Lampen)
            - Name der Lampe
            - x-Position
        - Liste der Output-Positionen (Lampen)
            - Name der Lampe
            - x-Position
    """
    with open(os.path.join(os.path.dirname(__file__), path), "r", encoding="utf8") as f:
        # Dimensionen der Konstruktion
        n, m = map(int, f.readline().split())

        board = np.empty((n, m), dtype=str)
        inp = []
        out = []

        for mi in range(m):
            for ni, c in enumerate(f.readline().split()[:n]):
                board[ni][mi] = c  # Charakter in `board` speichern
                if c.startswith("Q"):
                    inp.append((c, ni))  # mi ist 0, muss nicht gespeichert werden
                elif c.startswith("L"):
                    out.append((c, ni))  # mi ist m-1, muss nicht gespeichert werden

        assert len(inp) > 0, "Keine Lampe als Eingabe konfiguriert!"
        assert len(out) > 0, "Keine Lampe als Ausgabe konfiguriert!"
        return n, m, board, inp, out


# Bausteine werden durch lambda-Funktionen modelliert
kernels = {
    "rR": lambda a, b: (not b, not b),  # beide wenn Eingabe bei R 0 ist
    "Rr": lambda a, b: (not a, not a),  # beide wenn Eingabe bei R 0 ist
    "WW": lambda a, b: (not (a and b), not (a and b)),  # beide, solange nicht beide Eingaben 1 sind
    "BB": lambda a, b: (a, b),  # Eingabe wird weitergeleitet
}


def simulate(
    n: int,
    m: int,
    board: np.ndarray,
    inp: List[Tuple[str, int]],
    out: List[Tuple[str, int]],
    inp_states: tuple[bool, ...],
) -> Dict[str, bool]:
    """
    Simuliere die Konstruktion.

    Parameters
    ----------
    n : int
        Breite der Konstruktion.
    m : int
        Höhe der Konstruktion (Anzahl der Zeilen).
    board : np.ndarray
        2D-Array mit den Zeichen der Konstruktion.
    inp : List[Tuple[str, int]]
        Liste der Input-Positionen (Lampen)
            - Name der Lampe
            - x-Position
    out : List[Tuple[str, int]]
        Liste der Output-Positionen (Lampen)
            - Name der Lampe
            - x-Position
    inp_states : tuple[bool, ...]
        Eingabezustände.

    Returns
    -------
    Dict[str, bool]
        Ausgabezustände.
    """
    # Matrix für die Lichtzustände an einzelnen Positionen
    states = np.zeros((n, m), dtype=bool)
    # Eingabezustände setzen
    for inp, state in zip(inp, inp_states):
        states[inp[1]][0] = state
        # print(f'states[{inp[1]}][0] auf {state} gesetzt. ("{inp[0]}")')

    # Simulieren
    for mi in range(1, m - 1):  # für jede Zeile...
        ni = 0
        while True:  # ...wird von links durch die Positionen iteriert
            # überprüfen, ob es sich bei den nächsten beiden Zeichen um einen Baustein handelt
            kname = board[ni][mi] + board[ni + 1][mi]
            kernel = kernels.get(kname, None)
            if kernel:
                # Eingabezustände werden aus der Matrix gelesen
                a = states[ni][mi - 1]
                b = states[ni + 1][mi - 1]
                # Ausgabe des einzelnen Baustein wird berechnet
                c, d = kernel(a, b)
                # Ausgabezustände setzen
                states[ni][mi] = c
                states[ni + 1][mi] = d
                # die nächste Position wird übersprungen, da sie noch zum jetzigen Baustein gehört
                ni += 1
                # print(f"states[{ni}][{mi}] auf {c} und states[{ni+1}][{mi}] auf {d} gesetzt.")
            # else:
            #   print(f'kein kernel gefunden für "{kname}"')
            ni += 1
            if ni >= n - 1:
                break
    # Ausgabezustände auslesen und zurückgeben
    return {out[0]: states[out[1]][m - 2] for out in out}


def main(
    n: int,
    m: int,
    board: np.ndarray,
    inp: List[Tuple[str, int]],
    out: List[Tuple[str, int]],
    n_bsp: int,
):
    t_start = time.time()  # zeitmessung starten
    results = []
    for inp_states in product(
        [False, True], repeat=len(inp)
    ):  # für jede mögliche Kombination der Eingabe wird simuliert
        out_states = simulate(n, m, board, inp, out, inp_states)
        results.append(
            {
                **{inp[0]: inp_state for inp, inp_state in zip(inp, inp_states)},
                **out_states,
            }
        )

    results = pd.DataFrame(results)
    results = results.map(lambda x: "An" if x else "Aus")
    print(f"Simuliert in {((time.time() - t_start)*1000):.2f}ms")
    print()
    print(results)
    print()
    results.to_csv(f"output/nandu{n_bsp}.csv", index=False)
    print(f'Ausgabe gespeichert in "output/nandu{n_bsp}.csv"')


# Haupt-Loop
while True:
    try:
        n_bsp = int(input("Bitte Nummer des Beispiels eingeben:\n> "))
        n, m, board, inp, out = load_board(f"input/nandu{n_bsp}.txt")
        print()
        main(n, m, board, inp, out, n_bsp)
    except Exception as e:  # Error-Handling
        print(f"{e.__class__.__name__}: {e}")
    print()
