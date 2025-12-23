import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

# ---- Data ----
baseline_time = 1.566  # Infinity time

rows = [
    {"Method": "Infinity", "Time (s)": 1.566,   "Quality": 10.750, "Group": "Baseline"},
    {"Method": "FastVAR", "Time (s)": 0.649,    "Quality": 9.951,  "Group": "Competitor"},
    {"Method": "FastVAR†", "Time (s)": 0.698,  "Quality": 10.071, "Group": "Competitor"},
    {"Method": "ATP-VAR α=0.75", "Time (s)": 0.65158, "Quality": 10.350, "Group": "Ours"},
    {"Method": "ATP-VAR α=0.9",  "Time (s)": 0.6133,  "Quality": 10.150, "Group": "Ours"},
]

df = pd.DataFrame(rows)
df["Speedup"] = baseline_time / df["Time (s)"]

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
})

fig, ax = plt.subplots(figsize=(10, 6))

# default-cycle colors per group (no manual colors)
groups = list(df['Group'].unique())
group_to_color = {}
for g in groups:
    group_to_color[g] = ax._get_lines.get_next_color()

markers = {'Baseline': 'o', 'Competitor': 's', 'Ours': 'D'}

for g in groups:
    sub = df[df["Group"] == g]
    ax.scatter(
        sub["Speedup"], sub["Quality"],
        s=180, alpha=0.95,
        marker=markers.get(g, "o"),
        edgecolors="white", linewidths=1.2,
        label=g, zorder=5, color=group_to_color[g]
    )

# ---- Labels ----
for _, row in df.iterrows():
    label = (
        f"{row['Method']}\n"
        f"Speedup: {row['Speedup']:.2f}×\n"
        f"Quality: {row['Quality']:.3f}"
    )
    xytext = (10, 10)
    ha, va = "left", "bottom"

    if row["Method"] == "Infinity":
        xytext = (10, -35)
        ha, va = "left", "top"
    elif row["Method"] == "ATP-VAR α=0.75":
        xytext = (10, 28)   # up
    elif row["Method"] == "ATP-VAR α=0.9":
        xytext = (10, 20)  # down to avoid overlap
        ha, va = "left", "top"
    elif row["Method"] == "FastVAR":
        xytext = (-110, 0)
        ha, va = "left", "top"
    elif row["Method"] == "FastVAR†":
        xytext = (-120, 20)
        ha, va = "left", "top"
    ax.annotate(
        label,
        (row["Speedup"], row["Quality"]),
        textcoords="offset points",
        xytext=xytext,
        ha=ha, va=va,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85),
        zorder=10
    )

# ---- Pareto frontier (requested explicit path) ----
frontier_order = ["Infinity", "ATP-VAR α=0.75", "ATP-VAR α=0.9"]
frontier_df = df.set_index("Method").loc[frontier_order].reset_index()
ax.plot(
    frontier_df["Speedup"], frontier_df["Quality"],
    linestyle="-", linewidth=2, alpha=0.85,color="black",
    label="Pareto Frontier", zorder=3
)

# Arrow along the same path (Infinity -> α=0.75 -> α=0.9)
for a, b in zip(frontier_order[:-1], frontier_order[1:]):
    ra = df[df["Method"] == a].iloc[0]
    rb = df[df["Method"] == b].iloc[0]
    ax.annotate(
        "",
        xy=(rb["Speedup"], rb["Quality"]),
        xytext=(ra["Speedup"], ra["Quality"]),
        arrowprops=dict(arrowstyle="-", lw=2,alpha=0.85),
        zorder=4
    )

# ---- Axes: padding + ticks ----
xmin, xmax = df["Speedup"].min(), df["Speedup"].max()
ymin, ymax = df["Quality"].min(), df["Quality"].max()

xpad = (xmax - xmin) * 0.22 if xmax > xmin else 0.3
ypad = (ymax - ymin) * 0.30 if ymax > ymin else 0.2

ax.set_xlim(xmin - xpad, xmax + xpad)
ax.set_ylim(ymin - ypad, ymax + ypad)

ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}×"))
ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.2f}"))

ax.set_xlabel("Speedup (×)  [Higher is Better]")
ax.set_ylabel("Quality (HPSV3)  [Higher is Better]")
ax.set_title("Performance Comparison: Speedup vs Quality")

ax.grid(True, linestyle="--", alpha=0.35)
ax.legend(loc="lower left", frameon=True, framealpha=0.95)

fig.tight_layout()

out_path = "pareto_frontier_newdata_v3_frontier.png"
fig.savefig(out_path, dpi=300)
