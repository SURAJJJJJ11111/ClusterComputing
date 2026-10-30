# School / University Club Membership Network (PySpark)

Author: Suraj Kumar Prajapati | Roll No. 41 | Submitted to: Richa | Lovely Professional University

## What it does
1. Loads students, clubs and memberships from CSV (PySpark DataFrames)
2. Builds a bipartite graph (vertices = students + clubs, edges = memberships)
3. Finds the most popular club (club with the most student connections)
4. Finds students who are members of multiple clubs
5. Visualises the network (PNG + interactive HTML)

## Requirements
- Python 3.8 - 3.12
- Java 8 / 11 / 17 / 21 (needed by Spark) - check with `java -version`

## How to run
```
pip install -r requirements.txt
python main.py
```

## Results (in the `output` folder)
- `network.html`          interactive graph - double-click to open in a browser
- `network_dashboard.png` static picture of the network + popularity chart
- `console_output.txt`    full program output
- `club_popularity.csv`, `multi_club_students.csv`, `club_overlap.csv`

## Folder layout
```
data/        students.csv (250 x 8), clubs.csv (12 x 6), memberships.csv (438 x 6)
main.py      PySpark analysis + visualisation
dashboard_template.html   template used by main.py for network.html
report/      Word report with screenshots
```
Windows tip: if Spark complains about JAVA_HOME, install JDK 17 and set JAVA_HOME, then open a NEW terminal.
