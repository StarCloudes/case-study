import csv
import matplotlib.pyplot as plt
import numpy as np

# Read data from CSV
data = []
with open('scaling_results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        # Convert string values to appropriate types
        m = int(row['m'])
        n = int(row['n'])
        time = float(row['execution_time_seconds'])
        data.append({'m': m, 'n': n, 'time': time})

# Convert data to numpy structured arrays for easier slicing
m_values = sorted(set(d['m'] for d in data))
n_values = sorted(set(d['n'] for d in data))

# Create a mapping: for each fixed n, collect (m, time) pairs
data_by_n = {n: [] for n in n_values}
for d in data:
    data_by_n[d['n']].append((d['m'], d['time']))

# Plot execution time vs m for fixed n
plt.figure(figsize=(8, 6))
for n in sorted(data_by_n.keys()):
    # Sort by m for a consistent line
    pts = sorted(data_by_n[n], key=lambda x: x[0])
    ms, times = zip(*pts)
    plt.plot(ms, times, marker='o', label=f"n = {n}")
plt.xlabel("Number of Rows (m)")
plt.ylabel("Execution Time (seconds)")
plt.title("Scaling with Respect to m (Rows)")
plt.legend()
plt.grid(True)
plt.savefig("scaling_vs_m.png")
plt.show()

# Create a mapping: for each fixed m, collect (n, time) pairs
data_by_m = {m: [] for m in m_values}
for d in data:
    data_by_m[d['m']].append((d['n'], d['time']))

# Plot execution time vs n for fixed m
plt.figure(figsize=(8, 6))
for m in sorted(data_by_m.keys()):
    pts = sorted(data_by_m[m], key=lambda x: x[0])
    ns, times = zip(*pts)
    plt.plot(ns, times, marker='s', label=f"m = {m}")
plt.xlabel("Number of Columns (n)")
plt.ylabel("Execution Time (seconds)")
plt.title("Scaling with Respect to n (Columns)")
plt.legend()
plt.grid(True)
plt.savefig("scaling_vs_n.png")
plt.show()