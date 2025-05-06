import os
import pandas as pd
import matplotlib.pyplot as plt

# --- 设置文件夹（如需自定义） ---
log_dir = "./data/"  # 当前目录

# --- 加载所有日志文件到字典 ---
log_data = {}
for fname in os.listdir(log_dir):
    if fname.startswith("log_") and fname.endswith(".txt"):
        parts = fname.replace(".txt", "").split("_")
        try:
            N0 = int(parts[1][1:])
            lmax = int(parts[2][1:])
            nu = int(parts[3][2:])
        except ValueError:
            continue
        key = (N0, lmax, nu)
        df = pd.read_csv(os.path.join(log_dir, fname), names=["iter", "residual"])
        log_data[key] = df

# --- Experiment 1: Fixed N=128, vary lmax ---
plt.figure(figsize=(10, 6))
for lmax in [2, 3, 4, 5, 6]:
    key = (128, lmax, 10)  # nu = 10 fixed
    if key in log_data:
        df = log_data[key]
        plt.plot(df["iter"], df["residual"], label=f"lmax={lmax}")

plt.yscale("log")
plt.xlabel("V-cycle iteration")
plt.ylabel("Max residual")
plt.title("Experiment 1: Fixed N0=128, Vary lmax (nu=10)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("experiment1_vary_lmax.png")
plt.show()

# --- Experiment 2: Vary N0, fixed lmax=2 and lmax=6 ---
plt.figure(figsize=(10, 6))
for N0 in [16, 32, 64, 128, 256]:
    for lmax in [2, 6]:
        key = (N0, lmax, 10)
        if key in log_data:
            df = log_data[key]
            label = f"N0={N0}, lmax={lmax}"
            plt.plot(df["iter"], df["residual"], label=label)

plt.yscale("log")
plt.xlabel("V-cycle iteration")
plt.ylabel("Max residual")
plt.title("Experiment 2: Vary N0, Compare lmax=2 vs lmax=6 (nu=10)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("experiment2_vary_N0_lmax2_vs_lmax6.png")
plt.show()