#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 07: Trực quan hóa dữ liệu khám phá
ESG Bayesian Paper - AJEB
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# CUD colorblind-friendly palette (Color Universal Design)
CUD_PALETTE = [
    '#E69F00',  # Orange
    '#56B4E9',  # Sky Blue
    '#009E73',  # Bluish Green
    '#F0E442',  # Yellow
    '#0072B2',  # Blue
    '#D55E00',  # Vermillion
    '#CC79A7',  # Reddish Purple
    '#999999',  # Gray
]

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette(CUD_PALETTE)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Read data
df = pd.read_csv('data/processed/asean-esg-final.csv')
print(f"Loaded {len(df)} observations from {df['Economy'].nunique()} countries")
print(f"Year range: {df['year'].min()} - {df['year'].max()}")

# Define income groups for ASEAN countries (World Bank classification, approx 2020)
income_groups = {
    'Brunei Darussalam': 'High',
    'Singapore': 'High',
    'Malaysia': 'Upper-Middle',
    'Thailand': 'Upper-Middle',
    'Indonesia': 'Lower-Middle',
    'Philippines': 'Lower-Middle',
    'Vietnam': 'Lower-Middle',
    'Lao PDR': 'Lower-Middle',
    'Cambodia': 'Lower-Middle',
    'Myanmar': 'Lower-Middle',
}
df['Income_Group'] = df['Economy'].map(income_groups)

# Create output directory
Path('LaTeX/figures').mkdir(parents=True, exist_ok=True)

# === FIGURE 1: Time series ESG scores for ASEAN countries (faceted) ===
print("\n[1/4] Creating time series ESG plots...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# E_score
ax1 = axes[0]
for i, country in enumerate(df['Economy'].unique()):
    country_data = df[df['Economy'] == country].sort_values('year')
    ax1.plot(country_data['year'], country_data['E_score'], 
             label=country, color=CUD_PALETTE[i % len(CUD_PALETTE)], 
             linewidth=1.5, alpha=0.8)
ax1.set_ylabel('Chỉ số Môi trường (E)')
ax1.set_title('Xu hướng chỉ số ESG tại các nước ASEAN (1975-2024)', fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
ax1.grid(True, alpha=0.3)

# S_score
ax2 = axes[1]
for i, country in enumerate(df['Economy'].unique()):
    country_data = df[df['Economy'] == country].sort_values('year')
    ax2.plot(country_data['year'], country_data['S_score'], 
             label=country, color=CUD_PALETTE[i % len(CUD_PALETTE)], 
             linewidth=1.5, alpha=0.8)
ax2.set_ylabel('Chỉ số Xã hội (S)')
ax2.grid(True, alpha=0.3)

# G_score
ax3 = axes[2]
for i, country in enumerate(df['Economy'].unique()):
    country_data = df[df['Economy'] == country].sort_values('year')
    ax3.plot(country_data['year'], country_data['G_score'], 
             label=country, color=CUD_PALETTE[i % len(CUD_PALETTE)], 
             linewidth=1.5, alpha=0.8)
ax3.set_ylabel('Chỉ số Quản trị (G)')
ax3.set_xlabel('Năm')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('LaTeX/figures/fig-01-esg-time-series.png', dpi=300, bbox_inches='tight')
print("✓ Saved: LaTeX/figures/fig-01-esg-time-series.png")
plt.close()

# === FIGURE 2: Scatter plots ESG vs GDP growth ===
print("\n[2/4] Creating scatter plots ESG vs GDP growth...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# E_score vs GDP_growth
ax1 = axes[0]
df_clean = df.dropna(subset=['E_score', 'GDP_growth'])
scatter1 = ax1.scatter(df_clean['E_score'], df_clean['GDP_growth'], 
                       c=[CUD_PALETTE[i % len(CUD_PALETTE)] for i in range(len(df_clean))],
                       alpha=0.5, s=30)
ax1.set_xlabel('Chỉ số Môi trường (E)')
ax1.set_ylabel('Tăng trưởng GDP (%)')
ax1.set_title('Môi trường vs Tăng trưởng GDP', fontweight='bold')
ax1.grid(True, alpha=0.3)
# Add trend line
z = np.polyfit(df_clean['E_score'], df_clean['GDP_growth'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df_clean['E_score'].min(), df_clean['E_score'].max(), 100)
ax1.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8, label='Xu hướng')
ax1.legend()

# S_score vs GDP_growth
ax2 = axes[1]
df_clean = df.dropna(subset=['S_score', 'GDP_growth'])
scatter2 = ax2.scatter(df_clean['S_score'], df_clean['GDP_growth'], 
                       c=[CUD_PALETTE[i % len(CUD_PALETTE)] for i in range(len(df_clean))],
                       alpha=0.5, s=30)
ax2.set_xlabel('Chỉ số Xã hội (S)')
ax2.set_ylabel('Tăng trưởng GDP (%)')
ax2.set_title('Xã hội vs Tăng trưởng GDP', fontweight='bold')
ax2.grid(True, alpha=0.3)
z = np.polyfit(df_clean['S_score'], df_clean['GDP_growth'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df_clean['S_score'].min(), df_clean['S_score'].max(), 100)
ax2.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8, label='Xu hướng')
ax2.legend()

# G_score vs GDP_growth
ax3 = axes[2]
df_clean = df.dropna(subset=['G_score', 'GDP_growth'])
scatter3 = ax3.scatter(df_clean['G_score'], df_clean['GDP_growth'], 
                       c=[CUD_PALETTE[i % len(CUD_PALETTE)] for i in range(len(df_clean))],
                       alpha=0.5, s=30)
ax3.set_xlabel('Chỉ số Quản trị (G)')
ax3.set_ylabel('Tăng trưởng GDP (%)')
ax3.set_title('Quản trị vs Tăng trưởng GDP', fontweight='bold')
ax3.grid(True, alpha=0.3)
z = np.polyfit(df_clean['G_score'], df_clean['GDP_growth'], 1)
p = np.poly1d(z)
x_trend = np.linspace(df_clean['G_score'].min(), df_clean['G_score'].max(), 100)
ax3.plot(x_trend, p(x_trend), 'r--', linewidth=2, alpha=0.8, label='Xu hướng')
ax3.legend()

plt.tight_layout()
plt.savefig('LaTeX/figures/fig-02-scatter-esg-gdp.png', dpi=300, bbox_inches='tight')
print("✓ Saved: LaTeX/figures/fig-02-scatter-esg-gdp.png")
plt.close()

# === FIGURE 3: Box plots by income group ===
print("\n[3/4] Creating box plots by income group...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# E_score by income group
ax1 = axes[0]
income_order = ['High', 'Upper-Middle', 'Lower-Middle']
df_filtered = df[df['Income_Group'].notna()].copy()
df_filtered['Income_Group'] = pd.Categorical(df_filtered['Income_Group'], 
                                               categories=income_order, ordered=True)
sns.boxplot(data=df_filtered, x='Income_Group', y='E_score', 
            palette=[CUD_PALETTE[0], CUD_PALETTE[1], CUD_PALETTE[2]], ax=ax1)
ax1.set_xlabel('Nhóm thu nhập')
ax1.set_ylabel('Chỉ số Môi trường (E)')
ax1.set_title('Môi trường theo nhóm thu nhập', fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# S_score by income group
ax2 = axes[1]
sns.boxplot(data=df_filtered, x='Income_Group', y='S_score', 
            palette=[CUD_PALETTE[0], CUD_PALETTE[1], CUD_PALETTE[2]], ax=ax2)
ax2.set_xlabel('Nhóm thu nhập')
ax2.set_ylabel('Chỉ số Xã hội (S)')
ax2.set_title('Xã hội theo nhóm thu nhập', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# G_score by income group
ax3 = axes[2]
sns.boxplot(data=df_filtered, x='Income_Group', y='G_score', 
            palette=[CUD_PALETTE[0], CUD_PALETTE[1], CUD_PALETTE[2]], ax=ax3)
ax3.set_xlabel('Nhóm thu nhập')
ax3.set_ylabel('Chỉ số Quản trị (G)')
ax3.set_title('Quản trị theo nhóm thu nhập', fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('LaTeX/figures/fig-03-boxplot-income-group.png', dpi=300, bbox_inches='tight')
print("✓ Saved: LaTeX/figures/fig-03-boxplot-income-group.png")
plt.close()

# === FIGURE 4: Bar chart average ESG by country ===
print("\n[4/4] Creating bar chart average ESG by country...")

# Calculate average ESG by country
avg_esg = df.groupby('Economy')[['E_score', 'S_score', 'G_score']].mean().reset_index()
avg_esg['Economy'] = avg_esg['Economy'].str.replace(' ', '\n')  # Wrap long names

fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(avg_esg))
width = 0.25

bars1 = ax.bar(x - width, avg_esg['E_score'], width, 
               label='Môi trường (E)', color=CUD_PALETTE[0], alpha=0.8)
bars2 = ax.bar(x, avg_esg['S_score'], width, 
               label='Xã hội (S)', color=CUD_PALETTE[1], alpha=0.8)
bars3 = ax.bar(x + width, avg_esg['G_score'], width, 
               label='Quản trị (G)', color=CUD_PALETTE[2], alpha=0.8)

ax.set_xlabel('Quốc gia')
ax.set_ylabel('Điểm số trung bình')
ax.set_title('Điểm ESG trung bình theo quốc gia ASEAN', fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(avg_esg['Economy'], fontsize=8)
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

plt.tight_layout()
plt.savefig('LaTeX/figures/fig-04-bar-avg-esg-country.png', dpi=300, bbox_inches='tight')
print("✓ Saved: LaTeX/figures/fig-04-bar-avg-esg-country.png")
plt.close()

print("\n" + "="*60)
print("✅ Hoàn thành tạo tất cả 4 figures!")
print("="*60)
print("\nFiles created:")
print("  1. LaTeX/figures/fig-01-esg-time-series.png")
print("  2. LaTeX/figures/fig-02-scatter-esg-gdp.png")
print("  3. LaTeX/figures/fig-03-boxplot-income-group.png")
print("  4. LaTeX/figures/fig-04-bar-avg-esg-country.png")
print("\nAll figures use CUD colorblind-friendly palette.")
print("Resolution: 300 DPI (publication quality)")
