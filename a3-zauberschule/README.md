# Zauberschule

❔ A3 👥 00128 🧑 Leonhard Masche 📆 30.09.2023

## Lösungsidee

Zur Lösung dieses Shortest-Path-Problems gibt es einige bekannte Algorithmen. Die möglichen Wege in der Zauberschule können als gewichteter Graph dargestellt werden, wobei Bewegungen in die vier Richtungen (links, oben, rechts, unten) ein Gewicht von $1$, und Stockwerkwechsel ein Gewicht von $3$ haben. Um nun einen kürzesten Pfad zu finden, wird Dijkstra's Algorithmus verwendet. Für jeden besuchten Knoten (Feld) wird dessen Vorgänger gespeichert, sodass letztendlich der Pfad selbst zurückverfolgt werden kann. Entsprechend Dijktra's Algorithmus ist dies der kürzest mögliche Pfad.

## Umsetzung

Das Programm (`program.py`) ist in Python umgesetzt und mit einer Umgebung ab der Version $3.8$ ausführbar. Zum Umgang mit Matrizen wird die externe Bibliothek `numpy` verwendet.

Beim Ausführen der Datei wird zuerst nach der Zahl des Beispiels gefragt. Dieses wird nun aus der Datei `input/zauberschule{n}.txt` geladen und bearbeitet. Das Ergebnis wird, zusammen mit einigen Werten, ausgegeben. Das resultierende Zauberschule-Gitter wird zusätzlich in eine Datei geschrieben.

## Beispiele

Hier wird das Programm auf die <n> Beispiele von der Website angewendet:

``

```text

```

---


``

```text

```

---

`fname`

```text
output
```

## Quellcode

`program.py`

```python

```