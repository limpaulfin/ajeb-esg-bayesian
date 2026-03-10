#!/usr/bin/env python3
"""
Task 11: Phân tích posterior và credible intervals
- Posterior summary (mean, sd, HDI 94%)
- Probability of direction (pd)
- Forest plot
- HDI plots
- LaTeX table generation
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# Setup
PROJECT_ROOT = Path('.')
TRACE_PATH = PROJECT_ROOT / 'data/processed/trace.nc'
OUTPUT_DIR = PROJECT_ROOT / 'LaTeX' / 'figures'
TABLE_DIR = PROJECT_ROOT / 'LaTeX' / 'tables'

# CUD color palette (colorblind-friendly)
CUD_COLORS = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'red': '#CC79A7',
    'purple': '#9467BD',
    'brown': '#8C564B',
    'pink': '#F0E442',
    'gray': '#999999'
}

print("="*70)
print("TASK 11: PHÂN TÍCH POSTERIOR VÀ CREDIBLE INTERVALS")
print("="*70)

# === 1. LOAD TRACE ===
print("\n[1] Loading trace...")
trace = az.from_netcdf(TRACE_PATH)
print(f"✓ Trace loaded from: {TRACE_PATH}")
print(f"  Shape: {trace.posterior.dims}")

# === 2. POSTERIOR SUMMARY ===
print("\n[2] Computing posterior summary (HDI 94%)...")

params = ['alpha', 'beta_E', 'beta_S', 'beta_G', 'gamma_inflation', 'gamma_pop', 'sigma_u', 'sigma']

summary_df = az.summary(trace, var_names=params, hdi_prob=0.94)
print("✓ Summary computed")
print(summary_df)

# === 3. PROBABILITY OF DIRECTION (pd) ===
print("\n[3] Computing Probability of Direction (pd)...")

def compute_pd(samples):
    """
    Probability of Direction: proportion of posterior samples on the side of 0
    that is consistent with the sign of the mean.
    """
    mean_val = np.mean(samples)
    if mean_val >= 0:
        pd = np.mean(samples > 0)
    else:
        pd = np.mean(samples < 0)
    return pd

pd_results = {}
for param in params:
    samples = trace.posterior[param].values.flatten()
    pd = compute_pd(samples)
    pd_results[param] = pd
    print(f"  {param:20s}: pd = {pd:.4f} ({pd*100:.2f}%)")

# === 4. FOREST PLOT ===
print("\n[4] Creating forest plot...")

fig, ax = plt.subplots(figsize=(10, 6))

# Extract HDI for main coefficients
main_params = ['beta_E', 'beta_S', 'beta_G', 'gamma_inflation', 'gamma_pop']
param_labels = {
    'beta_E': r'$\beta_E$ (Environmental)',
    'beta_S': r'$\beta_S$ (Social)',
    'beta_G': r'$\beta_G$ (Governance)',
    'gamma_inflation': r'$\gamma_{inflation}$ (Lạm phát)',
    'gamma_pop': r'$\gamma_{pop}$ (Dân số)'
}

y_positions = range(len(main_params))
colors = [CUD_COLORS['blue'], CUD_COLORS['orange'], CUD_COLORS['green'], 
          CUD_COLORS['purple'], CUD_COLORS['red']]

for i, (param, color) in enumerate(zip(main_params, colors)):
    mean_val = summary_df.loc[param, 'mean']
    hdi_low = summary_df.loc[param, f'hdi_3%']
    hdi_high = summary_df.loc[param, f'hdi_97%']
    
    # Plot HDI as line
    ax.plot([hdi_low, hdi_high], [i, i], color=color, linewidth=3, zorder=2)
    
    # Plot mean as dot
    ax.scatter([mean_val], [i], color=color, s=100, zorder=3, edgecolor='black', linewidth=1)
    
    # Add text annotation
    pd_val = pd_results[param]
    significance = '***' if pd_val > 0.99 else '**' if pd_val > 0.95 else '*' if pd_val > 0.90 else ''
    ax.text(hdi_high + 0.01, i, f'{mean_val:.4f} [{hdi_low:.4f}, {hdi_high:.4f}] pd={pd_val:.2f}{significance}',
            va='center', ha='left', fontsize=9, family='monospace')

# Reference line at 0
ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)

ax.set_yticks(y_positions)
ax.set_yticklabels([param_labels[p] for p in main_params])
ax.set_xlabel('Giá trị hệ số (HDI 94%)', fontsize=12)
ax.set_title('Forest Plot: Hệ số hồi quy Bayesian với HDI 94%', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
ax.set_xlim(-0.2, 0.8)

plt.tight_layout()
forest_path = OUTPUT_DIR / 'fig-forest-plot.png'
plt.savefig(forest_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Forest plot saved: {forest_path}")
plt.close()

# === 5. HDI PLOTS FOR ESG PILLARS ===
print("\n[5] Creating HDI plots for ESG pillars...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

esg_params = ['beta_E', 'beta_S', 'beta_G']
esg_names = ['Environmental (E)', 'Social (S)', 'Governance (G)']
esg_colors = [CUD_COLORS['blue'], CUD_COLORS['orange'], CUD_COLORS['green']]

for idx, (param, name, color) in enumerate(zip(esg_params, esg_names, esg_colors)):
    ax = axes[idx]
    
    samples = trace.posterior[param].values.flatten()
    mean_val = summary_df.loc[param, 'mean']
    hdi_low = summary_df.loc[param, f'hdi_3%']
    hdi_high = summary_df.loc[param, f'hdi_97%']
    pd_val = pd_results[param]
    
    # Plot density
    sns.kdeplot(samples, ax=ax, color=color, linewidth=2, fill=True, alpha=0.3)
    
    # Add HDI region
    hdi_x = np.linspace(hdi_low, hdi_high, 100)
    kde = sns.kdeplot(samples, ax=ax, color=color).get_lines()[0].get_data()
    ax.fill_between(kde[0], kde[1], where=(kde[0] >= hdi_low) & (kde[0] <= hdi_high),
                    color=color, alpha=0.4, label='HDI 94%')
    
    # Mark mean
    ax.axvline(mean_val, color=color, linestyle='-', linewidth=2, label=f'Mean = {mean_val:.4f}')
    
    # Mark 0
    ax.axvline(0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Zero')
    
    # Annotate
    significance = '***' if pd_val > 0.99 else '**' if pd_val > 0.95 else '*' if pd_val > 0.90 else ''
    ax.text(0.95, 0.95, f'HDI: [{hdi_low:.4f}, {hdi_high:.4f}]\npd = {pd_val:.2%} {significance}',
            transform=ax.transAxes, va='top', ha='right', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'{name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Giá trị hệ số', fontsize=10)
    ax.set_ylabel('Mật độ', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(alpha=0.3, linestyle=':', linewidth=0.5)

plt.tight_layout()
hdi_path = OUTPUT_DIR / 'fig-hdi-esg-pillars.png'
plt.savefig(hdi_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ HDI plots saved: {hdi_path}")
plt.close()

# === 6. GENERATE LATEX TABLE ===
print("\n[6] Generating LaTeX table (Table 5)...")

latex_content = r"""\begin{table}[htbp]
\centering
\caption{Kết quả hồi quy Bayesian: Tác động của ESG đến tăng trưởng GDP}
\label{tab:bayesian-results}
\small
\begin{tabular}{lccccc}
\toprule
\textbf{Tham số} & \textbf{Mean} & \textbf{SD} & \textbf{HDI 94\%} & \textbf{pd} & \textbf{Kết luận} \\
\midrule
"""

for param in params:
    mean_val = summary_df.loc[param, 'mean']
    sd_val = summary_df.loc[param, 'sd']
    hdi_low = summary_df.loc[param, f'hdi_3%']
    hdi_high = summary_df.loc[param, f'hdi_97%']
    pd_val = pd_results[param]
    
    # Determine significance
    if pd_val > 0.99:
        sig = '***'
    elif pd_val > 0.95:
        sig = '**'
    elif pd_val > 0.90:
        sig = '*'
    else:
        sig = ''
    
    # Parameter label in Vietnamese
    if param == 'alpha':
        label = r'Intercept ($\alpha$)'
    elif param == 'beta_E':
        label = r'Environmental ($\beta_E$)'
    elif param == 'beta_S':
        label = r'Social ($\beta_S$)'
    elif param == 'beta_G':
        label = r'Governance ($\beta_G$)'
    elif param == 'gamma_inflation':
        label = r'Lạm phát ($\gamma_{inf}$)'
    elif param == 'gamma_pop':
        label = r'Tăng dân số ($\gamma_{pop}$)'
    elif param == 'sigma_u':
        label = r'Random effects ($\sigma_u$)'
    elif param == 'sigma':
        label = r'Residual ($\sigma$)'
    
    # Conclusion based on HDI and pd
    if param in ['beta_E', 'beta_S', 'beta_G']:
        if hdi_low > 0:
            conclusion = 'Dương có ý nghĩa'
        elif hdi_high < 0:
            conclusion = 'Âm có ý nghĩa'
        elif pd_val > 0.90:
            direction = 'dương' if mean_val > 0 else 'âm'
            conclusion = f'Xu hướng {direction} yếu'
        else:
            conclusion = 'Không có ý nghĩa'
    else:
        conclusion = '-'
    
    latex_content += f"{label} & {mean_val:.4f} & {sd_val:.4f} & [{hdi_low:.4f}, {hdi_high:.4f}] & {pd_val:.2%} {sig} & {conclusion} \\\\\n"

latex_content += r"""\midrule
\multicolumn{6}{l}{\textit{Ghi chú:} HDI = Highest Density Interval. pd = Probability of Direction.} \\
\multicolumn{6}{l}{\textit{Significance:} *** pd > 99\%, ** pd > 95\%, * pd > 90\%.} \\
\bottomrule
\end{tabular}
\end{table}
"""

table_path = TABLE_DIR / 'table-05-bayesian-results.tex'
with open(table_path, 'w', encoding='utf-8') as f:
    f.write(latex_content)

print(f"✓ LaTeX table saved: {table_path}")

# === 7. SAVE SUMMARY CSV ===
print("\n[7] Saving summary CSV...")

summary_output = summary_df.copy()
summary_output['pd'] = summary_output.index.map(pd_results)
summary_output['param'] = summary_output.index

csv_path = PROJECT_ROOT / 'data' / 'processed' / 'posterior-summary.csv'
summary_output.to_csv(csv_path, index=True)
print(f"✓ Summary CSV saved: {csv_path}")

# === 8. INTERPRETATION ===
print("\n[8] Interpretation:")
print("="*70)

print("\nCác chỉ số ESG chính:")
for param in ['beta_E', 'beta_S', 'beta_G']:
    mean_val = summary_df.loc[param, 'mean']
    hdi_low = summary_df.loc[param, f'hdi_3%']
    hdi_high = summary_df.loc[param, f'hdi_97%']
    pd_val = pd_results[param]
    
    if param == 'beta_E':
        pillar = 'Environmental (Môi trường)'
    elif param == 'beta_S':
        pillar = 'Social (Xã hội)'
    else:
        pillar = 'Governance (Quản trị)'
    
    print(f"\n{pillar}:")
    print(f"  - Hệ số: {mean_val:.4f} [{hdi_low:.4f}, {hdi_high:.4f}]")
    print(f"  - Probability of Direction: {pd_val:.2%}")
    
    if hdi_low > 0:
        print(f"  → Tác động DƯƠNG có ý nghĩa đến GDP (HDI không chứa 0)")
    elif hdi_high < 0:
        print(f"  → Tác động ÂM có ý nghĩa đến GDP (HDI không chứa 0)")
    elif pd_val > 0.90:
        direction = "dương" if mean_val > 0 else "âm"
        print(f"  → Xu hướng {direction} nhưng chưa có ý nghĩa (HDI chứa 0)")
    else:
        print(f"  → KHÔNG có tác động có ý nghĩa (pd < 90%)")

print("\nBiến kiểm soát:")
for param in ['gamma_inflation', 'gamma_pop']:
    mean_val = summary_df.loc[param, 'mean']
    hdi_low = summary_df.loc[param, f'hdi_3%']
    hdi_high = summary_df.loc[param, f'hdi_97%']
    pd_val = pd_results[param]
    
    if param == 'gamma_inflation':
        var = 'Lạm phát'
    else:
        var = 'Tăng dân số'
    
    print(f"\n{var}:")
    print(f"  - Hệ số: {mean_val:.4f} [{hdi_low:.4f}, {hdi_high:.4f}]")
    print(f"  - Probability of Direction: {pd_val:.2%}")

print("\n" + "="*70)
print("✅ TASK 11 COMPLETED")
print("="*70)
print("\nOutput files:")
print(f"  1. {forest_path}")
print(f"  2. {hdi_path}")
print(f"  3. {table_path}")
print(f"  4. {csv_path}")
