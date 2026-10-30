"""
School / University Club Memberships - Bipartite Graph Analysis with PySpark
Author : Suraj Kumar Prajapati (Roll No. 41)

What this program does
  1. Loads students, clubs and memberships from CSV files (PySpark DataFrames)
  2. Builds a bipartite graph  ->  vertices = students + clubs, edges = memberships
  3. Finds the most popular club (club vertex with the highest degree)
  4. Finds students who are members of multiple clubs (student vertices with degree > 1)
  5. Visualises the network (PNG dashboard + interactive HTML page)

Run:  python main.py
"""
import json
import os
import sys

os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(BASE, "output")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- Spark session
spark = (SparkSession.builder.appName("ClubMembershipGraph").master("local[*]")
         .config("spark.ui.enabled", "false")
         .config("spark.sql.shuffle.partitions", "4")
         .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")

line = "=" * 64

# ---------------------------------------------------------------- 1. Load dataset
students = spark.read.csv(os.path.join(DATA, "students.csv"), header=True, inferSchema=True)
clubs = spark.read.csv(os.path.join(DATA, "clubs.csv"), header=True, inferSchema=True)
members = spark.read.csv(os.path.join(DATA, "memberships.csv"), header=True, inferSchema=True)

print(line)
print(" 1. DATASET LOADED")
print(line)
print(f"students.csv    : {students.count():>4} rows x {len(students.columns)} columns")
print(f"clubs.csv       : {clubs.count():>4} rows x {len(clubs.columns)} columns")
print(f"memberships.csv : {members.count():>4} rows x {len(members.columns)} columns")
students.show(5, truncate=False)

# ---------------------------------------------------------------- 2. Bipartite graph
# Vertices : every student and every club (a 'type' column keeps the two sides apart)
# Edges    : one edge per membership  (student  --member_of-->  club)
v_students = students.select(F.col("student_id").alias("id"), F.col("name"), F.lit("student").alias("type"))
v_clubs = clubs.select(F.col("club_id").alias("id"), F.col("club_name").alias("name"), F.lit("club").alias("type"))
vertices = v_students.unionByName(v_clubs)
edges = members.select(F.col("student_id").alias("src"), F.col("club_id").alias("dst"),
                       F.lit("member_of").alias("relationship"), "role", "attendance_pct", "active")

n_students, n_clubs = v_students.count(), v_clubs.count()
n_edges = edges.count()
print(line)
print(" 2. BIPARTITE GRAPH CONSTRUCTED")
print(line)
print(f"Vertices : {vertices.count()}  ({n_students} students + {n_clubs} clubs)")
print(f"Edges    : {n_edges}  (student -> club memberships)")

# Degree of every vertex = number of edges touching it
club_degree = edges.groupBy("dst").agg(F.count("*").alias("members"),
                                       F.round(F.avg("attendance_pct"), 1).alias("avg_attendance"))
student_degree = edges.groupBy("src").agg(F.count("*").alias("num_clubs"))

# ---------------------------------------------------------------- 3. Most popular club
popularity = (club_degree.join(clubs, club_degree.dst == clubs.club_id)
              .select("club_name", "category", "members", "avg_attendance")
              .orderBy(F.desc("members")))
top_count = popularity.agg(F.max("members")).first()[0]
top_clubs = popularity.filter(F.col("members") == top_count).collect()

print(line)
print(" 3. MOST POPULAR CLUB")
print(line)
for r in top_clubs:
    print(f"Most popular club: {r['club_name']} with {r['members']} members")
print("\nClub popularity ranking:")
popularity.show(truncate=False)

# ---------------------------------------------------------------- 4. Students in multiple clubs
multi = (student_degree.filter(F.col("num_clubs") > 1)
         .join(edges.join(clubs, edges.dst == clubs.club_id)
               .groupBy("src").agg(F.concat_ws(", ", F.sort_array(F.collect_list("club_name"))).alias("clubs")),
               "src")
         .join(students, student_degree.src == students.student_id)
         .select("student_id", "name", "program", "num_clubs", "clubs")
         .orderBy(F.desc("num_clubs"), "student_id"))
n_multi = multi.count()
no_club = n_students - student_degree.count()

print(line)
print(" 4. STUDENTS WHO ARE MEMBERS OF MULTIPLE CLUBS")
print(line)
print(f"{n_multi} of {n_students} students are in more than one club.")
print("\nTop 15 (most clubs first):")
multi.show(15, truncate=False)

dist = (student_degree.groupBy("num_clubs").agg(F.count("*").alias("students")).orderBy("num_clubs"))
print("Clubs-per-student distribution:")
dist.show()
print(f"Students with no club: {no_club}")

# ---------------------------------------------------------------- bonus: club overlap
a, b = edges.alias("a"), edges.alias("b")
overlap = (a.join(b, (F.col("a.src") == F.col("b.src")) & (F.col("a.dst") < F.col("b.dst")))
           .groupBy(F.col("a.dst").alias("club1"), F.col("b.dst").alias("club2"))
           .agg(F.count("*").alias("shared_students"))
           .join(clubs.select(F.col("club_id").alias("club1"), F.col("club_name").alias("name1")), "club1")
           .join(clubs.select(F.col("club_id").alias("club2"), F.col("club_name").alias("name2")), "club2")
           .select("name1", "name2", "shared_students").orderBy(F.desc("shared_students")))
print(line)
print(" 5. BONUS: CLUB PAIRS SHARING THE MOST STUDENTS")
print(line)
overlap.show(5, truncate=False)

# ---------------------------------------------------------------- save results (CSV)
popularity.toPandas().to_csv(os.path.join(OUT, "club_popularity.csv"), index=False)
multi.toPandas().to_csv(os.path.join(OUT, "multi_club_students.csv"), index=False)
overlap.toPandas().to_csv(os.path.join(OUT, "club_overlap.csv"), index=False)

# ---------------------------------------------------------------- 5. Visualisation
vp = vertices.toPandas()
ep = edges.select("src", "dst").toPandas()
deg_students = student_degree.toPandas().set_index("src")["num_clubs"].to_dict()
deg_clubs = club_degree.toPandas().set_index("dst")["members"].to_dict()
spark.stop()

name = dict(zip(vp["id"], vp["name"]))
G = nx.Graph()
for _, r in vp.iterrows():
    G.add_node(r["id"], bipartite=0 if r["type"] == "student" else 1, kind=r["type"])
G.add_edges_from(ep.itertuples(index=False, name=None))
club_ids = [n for n, d in G.nodes(data=True) if d["kind"] == "club"]
stud_ids = [n for n, d in G.nodes(data=True) if d["kind"] == "student"]
isolated = [n for n in stud_ids if G.degree(n) == 0]
G.remove_nodes_from(isolated)               # students with no club are left out of the picture
stud_ids = [n for n in stud_ids if n in G]
best = top_clubs[0]["club_name"]
best_id = [k for k, v in name.items() if v == best][0]

# Layout: clubs on a ring, students pulled between the clubs they belong to
ring = nx.circular_layout(club_ids, scale=1.0)
pos = nx.spring_layout(G, pos=ring, fixed=club_ids, k=0.28, iterations=200, seed=41)

palette = {1: "#8fbfee", 2: "#378add", 3: "#1b4f8a", 4: "#0b2a52"}
fig, axes = plt.subplots(1, 2, figsize=(19, 9), gridspec_kw={"width_ratios": [2.1, 1]})
ax = axes[0]
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.18, width=0.6, edge_color="#666666")
nx.draw_networkx_nodes(G, pos, nodelist=stud_ids, ax=ax, node_size=[40 + 22 * G.degree(n) for n in stud_ids],
                       node_color=[palette[min(G.degree(n), 4)] for n in stud_ids], linewidths=0)
nx.draw_networkx_nodes(G, pos, nodelist=club_ids, ax=ax, node_size=[300 + 28 * deg_clubs.get(n, 0) for n in club_ids],
                       node_color=["#1d9e75" if n == best_id else "#d85a30" for n in club_ids],
                       edgecolors="white", linewidths=1.5)
import matplotlib.patheffects as pe
for n in club_ids:      # club names drawn with a white outline so they stay readable
    ax.text(pos[n][0], pos[n][1], name[n].replace(" Club", ""), ha="center", va="center", fontsize=10,
            fontweight="bold", color="#111111", zorder=10,
            path_effects=[pe.withStroke(linewidth=3.5, foreground="white")])
ax.set_title("Student - Club Membership Network (bipartite graph)", fontsize=15, fontweight="bold")
ax.axis("off")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#1d9e75", ms=12, label=f"Most popular club ({best})"),
                   Line2D([], [], marker="o", ls="", color="#d85a30", ms=12, label="Other clubs"),
                   Line2D([], [], marker="o", ls="", color=palette[1], ms=7, label="Student - 1 club"),
                   Line2D([], [], marker="o", ls="", color=palette[2], ms=8, label="Student - 2 clubs"),
                   Line2D([], [], marker="o", ls="", color=palette[3], ms=9, label="Student - 3 clubs"),
                   Line2D([], [], marker="o", ls="", color=palette[4], ms=10, label="Student - 4 clubs")],
          loc="lower left", fontsize=9, frameon=True)

import pandas as pd
pop = pd.read_csv(os.path.join(OUT, "club_popularity.csv")).sort_values("members")
ax2 = axes[1]
ax2.barh(pop["club_name"], pop["members"], color=["#1d9e75" if n == best else "#d85a30" for n in pop["club_name"]])
for i, v in enumerate(pop["members"]):
    ax2.text(v + 0.5, i, str(v), va="center", fontsize=10)
ax2.set_title("Club popularity (number of members)", fontsize=15, fontweight="bold")
ax2.set_xlabel("Members")
ax2.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "network_dashboard.png"), dpi=110)
plt.close()

# Interactive, fully offline HTML (hover / tap a node to highlight its connections)
xs = [pos[n][0] for n in G]; ys = [pos[n][1] for n in G]
minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
W, H, PAD = 900, 700, 40
def sx(x): return round(PAD + (x - minx) / (maxx - minx) * (W - 2 * PAD), 1)
def sy(y): return round(PAD + (y - miny) / (maxy - miny) * (H - 2 * PAD), 1)
payload = {
    "best": best,
    "nodes": [{"id": n, "name": name[n], "kind": G.nodes[n]["kind"], "x": sx(pos[n][0]), "y": sy(pos[n][1]),
               "deg": G.degree(n)} for n in G],
    "edges": [[u, v] for u, v in G.edges()],
    "stats": {"students": n_students, "clubs": n_clubs, "memberships": n_edges, "multi": n_multi},
    "ranking": pop.sort_values("members", ascending=False)[["club_name", "members"]].values.tolist(),
}
html = open(os.path.join(BASE, "dashboard_template.html"), encoding="utf-8").read()
html = html.replace("__DATA__", json.dumps(payload))
open(os.path.join(OUT, "network.html"), "w", encoding="utf-8").write(html)

print(line)
print(" 6. VISUALISATION SAVED")
print(line)
print("output/network_dashboard.png  (static picture)")
print("output/network.html           (interactive - open in your browser)")
print("output/club_popularity.csv, multi_club_students.csv, club_overlap.csv")
