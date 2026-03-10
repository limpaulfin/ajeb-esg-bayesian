#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 05: Ma trận tương quan và đa cộng tuyến
AJEB ESG Bayesian Paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings
warnings.filterwarnings('ignore')

# Configuration
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Read data
print("Đang đọc dữ liệu...")
df = pd.read_csv('data/processed/asean-esg-final.csv')
print(f"Số quan sát: {len(df)}")
print(f"Các biến: {df.columns.tolist()}")

# Define variables
dependent_var = 'GDP_growth'
independent_vars = ['E_score', 'S_score', 'G_score', 'inflation', 'population_growth']
all_vars = [dependent_var] + independent_vars

# 1. Correlation Matrix (Pearson)
print("\n1. Tính ma trận tương quan Pearson...")
corr_pearson = df[all_vars].corr(method='pearson')
print(corr_pearson.round(3))

# 2. Correlation Matrix (Spearman)
print("\n2. Tính ma trận tương quan Spearman...")
corr_spearman = df[all_vars].corr(method='spearman')
print(corr_spearman.round(3))

# 3. VIF Calculation
print("\n3. Tính VIF cho các biến độc lập...")
# Remove rows with missing values for VIF
df_vif = df[independent_vars].dropna()

if len(df_vif) > 0:
    # Add constant for VIF calculation
    X = add_constant(df_vif)
    
    vif_data = pd.DataFrame()
    vif_data['Variable'] = independent_vars
    vif_data['VIF'] = [variance_inflation_factor(X.values, i+1) for i in range(len(independent_vars))]
    
    print("\nVIF Results:")
    print(vif_data.to_string(index=False))
    
    # Check multicollinearity
    high_vif = vif_data[vif_data['VIF'] > 10]
    if len(high_vif) > 0:
        print(f"\n⚠️ CẢNH BÁO: Phát hiện đa cộng tuyến nghiêm trọng (VIF > 10):")
        print(high_vif.to_string(index=False))
    else:
        print("\n✓ Không có đa cộng tuyến nghiêm trọng (VIF < 10 cho tất cả biến)")
else:
    print("⚠️ Không đủ dữ liệu để tính VIF")

# 4. Correlation Heatmap
print("\n4. Tạo correlation heatmap...")
fig, ax = plt.subplots(figsize=(10, 8))

# Create mask for upper triangle
mask = np.triu(np.ones_like(corr_pearson, dtype=bool))

# Custom color palette (CUD palette for accessibility)
cmap = sns.diverging_palette(220, 20, as_cmap=True)

# Draw heatmap
sns.heatmap(corr_pearson, 
            mask=mask,
            annot=True, 
            fmt='.3f', 
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Hệ số tương quan Pearson"},
            vmin=-1, 
            vmax=1,
            ax=ax)

ax.set_title('Hình 1: Ma trận tương quan giữa các biến trong mô hình', 
             fontsize=14, pad=20, fontweight='bold')

plt.tight_layout()
plt.savefig('LaTeX/figures/fig-correlation-heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Đã lưu heatmap: LaTeX/figures/fig-correlation-heatmap.png")

# 5. Generate LaTeX Table
print("\n5. Tạo bảng LaTeX...")
latex_output = []
latex_output.append("\\begin{table}[htbp]")
latex_output.append("\\centering")
latex_output.append("\\caption{Ma trận tương quan Pearson}")
latex_output.append("\\label{tab:correlation}")
latex_output.append("\\small")
latex_output.append("\\begin{tabular}{l" + "r" * len(all_vars) + "}")
latex_output.append("\\hline")
latex_output.append("\\hline")

# Header row
header = "Biến & " + " & ".join([v.replace('_', '\\_') for v in all_vars]) + " \\\\"
latex_output.append(header)
latex_output.append("\\hline")

# Data rows
for i, var in enumerate(all_vars):
    row_values = [f"{corr_pearson.loc[var, v]:.3f}" for v in all_vars]
    row = var.replace('_', '\\_') + " & " + " & ".join(row_values) + " \\\\"
    latex_output.append(row)

latex_output.append("\\hline")
latex_output.append("\\hline")
latex_output.append("\\end{tabular}")
latex_output.append("\\begin{tablenotes}")
latex_output.append("\\small")
latex_output.append("\\item Ghi chú: N = 563 quan sát")
latex_output.append("\\end{tablenotes}")
latex_output.append("\\end{table}")

latex_table = "\n".join(latex_output)
print(latex_table)

with open('LaTeX/tables/table-02-correlation.tex', 'w', encoding='utf-8') as f:
    f.write(latex_table)
print("\n✓ Đã lưu bảng LaTeX: LaTeX/tables/table-02-correlation.tex")

# 6. Summary Statistics
print("\n6. Thống kê tóm tắt cho các biến...")
summary_stats = df[all_vars].describe()
print(summary_stats.round(3))

# 7. Save results to markdown report
print("\n7. Tạo báo cáo markdown...")
report = f"""# Task 05: Ma trận tương quan và đa cộng tuyến

## Kết quả phân tích

### 1. Ma trận tương quan Pearson

{corr_pearson.round(3).to_markdown()}

### 2. Ma trận tương quan Spearman

{corr_spearman.round(3).to_markdown()}

### 3. Variance Inflation Factor (VIF)

{vif_data.to_markdown(index=False)}

**Đánh giá đa cộng tuyến:**
"""

if len(high_vif) > 0:
    report += f"- ⚠️ CẢNH BÁO: Phát hiện {len(high_vif)} biến có VIF > 10\n"
    for _, row in high_vif.iterrows():
        report += f"  - {row['Variable']}: VIF = {row['VIF']:.2f}\n"
else:
    report += "- ✓ Không có đa cộng tuyến nghiêm trọng (VIF < 10 cho tất cả biến)\n"

report += f"""
### 4. Các phát hiện chính

**Tương quan với biến phụ thuộc (GDP_growth):**
"""

for var in independent_vars:
    corr_val = corr_pearson.loc['GDP_growth', var]
    direction = "dương" if corr_val > 0 else "âm"
    strength = abs(corr_val)
    if strength < 0.3:
        strength_text = "yếu"
    elif strength < 0.7:
        strength_text = "trung bình"
    else:
        strength_text = "mạnh"
    
    report += f"- {var}: r = {corr_val:.3f} (tương quan {direction}, {strength_text})\n"

report += f"""
### 5. Tương quan giữa các biến độc lập

**Tương quan cao giữa các biến ESG:**
- E_score vs S_score: r = {corr_pearson.loc['E_score', 'S_score']:.3f}
- E_score vs G_score: r = {corr_pearson.loc['E_score', 'G_score']:.3f}
- S_score vs G_score: r = {corr_pearson.loc['S_score', 'G_score']:.3f}

### 6. Output files

1. **LaTeX/tables/table-02-correlation.tex** - Bảng tương quan Pearson
2. **LaTeX/figures/fig-correlation-heatmap.png** - Heatmap tương quan
3. **code/task-05-correlation-vif.py** - Code Python

---

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Task:** 05 - Ma trận tương quan và đa cộng tuyến
"""

with open('docs/task-05-report.md', 'w', encoding='utf-8') as f:
    f.write(report)
print("✓ Đã lưu báo cáo: docs/task-05-report.md")

print("\n" + "="*60)
print("HOÀN THÀNH TASK 05")
print("="*60)
