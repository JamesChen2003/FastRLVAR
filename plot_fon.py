import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

original_time = 2.168098
data = {
    'Method': ['Original', 'FastVar', 'Our (Fast)', 'Our (Quality)'],
    'Time (s)': [2.168098, 0.814935, 0.642997+ 0.009, 0.683150+ 0.009],
    'HPSV3': [10.750, 10.071, 10.154, 10.278],
    'Group': ['Baseline', 'Competitor', 'Ours', 'Ours']
}

df = pd.DataFrame(data)
df['Speedup'] = original_time / df['Time (s)']

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

colors = {'Baseline': 'gray', 'Competitor': 'blue', 'Ours': '#d62728'}
markers = {'Baseline': 'o', 'Competitor': 's', 'Ours': 'D'}

for group in df['Group'].unique():
    subset = df[df['Group'] == group]
    plt.scatter(subset['Speedup'], subset['HPSV3'],
                label=group,
                color=colors[group],
                marker=markers[group],
                s=150, edgecolors='k', zorder=5)

for i, row in df.iterrows():
    xytext = (5, 5)
    ha = 'left'
    va = 'bottom'

    if row['Method'] == 'FastVar':
        xytext = (0, -25)
        ha = 'center'
    elif row['Method'] == 'Our (Fast)':
        xytext = (-5, 12)
        ha = 'center'
    elif row['Method'] == 'Our (Quality)':
        xytext = (5, 5)
    elif row['Method'] == 'Original':
        xytext = (-10, -20)
        ha = 'right'

    plt.annotate(row['Method'],
                 (row['Speedup'], row['HPSV3']),
                 xytext=xytext, textcoords='offset points',
                 fontsize=12, fontweight='bold', ha=ha, va=va)

frontier_methods = ['Original', 'Our (Quality)', 'Our (Fast)']
frontier_df = df[df['Method'].isin(frontier_methods)].sort_values('Speedup')

plt.plot(frontier_df['Speedup'], frontier_df['HPSV3'],
         color='#d62728', linestyle='--', alpha=0.6, linewidth=2, label='Pareto Frontier')

fastvar = df[df['Method'] == 'FastVar'].iloc[0]
our_fast = df[df['Method'] == 'Our (Fast)'].iloc[0]
our_quality = df[df['Method'] == 'Our (Quality)'].iloc[0]
plt.annotate("",
             xy=(our_quality['Speedup'], our_quality['HPSV3']),
             xytext=(fastvar['Speedup'], fastvar['HPSV3']),
             arrowprops=dict(arrowstyle="->", color="green", lw=2, ls='-'))
plt.xlabel('Speedup vs Original (x) [Higher is Better]', fontsize=13)
plt.ylabel('HPSV3 Score [Higher is Better]', fontsize=13)
plt.title('Performance Comparison: Speedup vs Quality', fontsize=15, fontweight='bold', y=1.02)

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower left', fontsize=11, frameon=True, framealpha=0.9)

plt.tight_layout()
plt.savefig('pareto_frontier_clean.png', dpi=300)