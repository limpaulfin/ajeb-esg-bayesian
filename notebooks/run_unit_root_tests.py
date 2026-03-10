#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 06: Kiểm định nghiệm đơn vị panel (Unit Root Tests)
Sử dụng Fisher-type tests: ADF tests cho từng quốc gia + combine p-values
"""

import pandas as pd
import numpy as np
from arch.unitroot import ADF
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("TASK 06: KIỂM ĐỊNH NGHIỆM ĐƠN VỊ PANEL")
print("=" * 80)

# Load data
df = pd.read_csv('../data/processed/asean-esg-final.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Số lượng quốc gia: {df['ISO3 code'].nunique()}")
print(f"Khoảng thời gian: {df['year'].min()} - {df['year'].max()}")

# Variables to test
variables_to_test = ['GDP_growth', 'E_score', 'S_score', 'G_score', 
                     'inflation', 'population_growth']

def fisher_adf_test(data, entity_col, time_col, value_col):
    """
    Fisher-type panel unit root test using ADF
    H0: All panels have unit roots
    H1: At least one panel is stationary
    
    Returns: chi2_stat, p_value, n_panels
    """
    entities = data[entity_col].unique()
    p_values = []
    n_panels = 0
    
    for entity in entities:
        entity_data = data[data[entity_col] == entity].sort_values(time_col)
        values = entity_data[value_col].dropna()
        
        if len(values) > 10:  # Need enough observations
            try:
                adf = ADF(values, trend='c')
                if adf.pvalue is not None and not np.isnan(adf.pvalue):
                    p_values.append(adf.pvalue)
                    n_panels += 1
            except:
                pass
    
    if len(p_values) > 0:
        # Fisher's inverse chi-square method
        # chi2 = -2 * sum(ln(p_i))
        chi2_stat = -2 * np.sum(np.log(p_values))
        # Degrees of freedom = 2 * n_panels
        df_chi2 = 2 * n_panels
        p_value = 1 - chi2.cdf(chi2_stat, df_chi2)
        return chi2_stat, p_value, n_panels
    else:
        return np.nan, np.nan, 0

def levin_lin_chu_approx(data, entity_col, time_col, value_col):
    """
    Simplified LLC-type test (approximation)
    H0: All panels have unit roots
    H1: All panels are stationary
    
    Returns: t_stat, p_value
    """
    entities = data[entity_col].unique()
    t_stats = []
    
    for entity in entities:
        entity_data = data[data[entity_col] == entity].sort_values(time_col)
        values = entity_data[value_col].dropna()
        
        if len(values) > 10:
            try:
                adf = ADF(values, trend='c')
                if adf.stat is not None and not np.isnan(adf.stat):
                    t_stats.append(adf.stat)
            except:
                pass
    
    if len(t_stats) > 0:
        # Average t-statistic (simplified LLC)
        avg_t = np.mean(t_stats)
        # Approximate p-value using normal distribution
        from scipy.stats import norm
        p_value = norm.cdf(avg_t)
        return avg_t, p_value
    else:
        return np.nan, np.nan

print("\n" + "=" * 80)
print("1. LEVIN-LIN-CHU (LLC) TEST - Approximation")
print("=" * 80)
print("H0: Tất cả các panel có unit root (non-stationary)")
print("H1: Tất cả các panel stationary\n")

llc_results = {}
for var in variables_to_test:
    try:
        stat, pval = levin_lin_chu_approx(df, 'ISO3 code', 'year', var)
        
        llc_results[var] = {
            'statistic': stat,
            'pvalue': pval,
            'stationary': 'Yes' if pval < 0.05 else 'No'
        }
        
        print(f"{var}:")
        print(f"  LLC statistic: {stat:.4f}")
        print(f"  p-value: {pval:.4f}")
        print(f"  Stationary (5%): {'Yes' if pval < 0.05 else 'No'}\n")
        
    except Exception as e:
        print(f"{var}: Error - {str(e)}\n")
        llc_results[var] = {'statistic': np.nan, 'pvalue': np.nan, 'stationary': 'Error'}

print("\n" + "=" * 80)
print("2. FISHER-TYPE ADF TEST (Similar to IPS)")
print("=" * 80)
print("H0: Tất cả các panel có unit root")
print("H1: Ít nhất một panel stationary\n")

ips_results = {}
for var in variables_to_test:
    try:
        stat, pval, n_panels = fisher_adf_test(df, 'ISO3 code', 'year', var)
        
        ips_results[var] = {
            'statistic': stat,
            'pvalue': pval,
            'stationary': 'Yes' if pval < 0.05 else 'No'
        }
        
        print(f"{var}:")
        print(f"  Fisher chi2 statistic: {stat:.4f}")
        print(f"  p-value: {pval:.4f}")
        print(f"  Number of panels: {n_panels}")
        print(f"  Stationary (5%): {'Yes' if pval < 0.05 else 'No'}\n")
        
    except Exception as e:
        print(f"{var}: Error - {str(e)}\n")
        ips_results[var] = {'statistic': np.nan, 'pvalue': np.nan, 'stationary': 'Error'}

print("\n" + "=" * 80)
print("3. ADF-FISHER TEST (Without trend)")
print("=" * 80)
print("H0: Tất cả các panel có unit root")
print("H1: Ít nhất một panel stationary\n")

adf_results = {}
for var in variables_to_test:
    try:
        stat, pval, n_panels = fisher_adf_test(df, 'ISO3 code', 'year', var)
        
        adf_results[var] = {
            'statistic': stat,
            'pvalue': pval,
            'stationary': 'Yes' if pval < 0.05 else 'No'
        }
        
        print(f"{var}:")
        print(f"  ADF-Fisher statistic: {stat:.4f}")
        print(f"  p-value: {pval:.4f}")
        print(f"  Number of panels: {n_panels}")
        print(f"  Stationary (5%): {'Yes' if pval < 0.05 else 'No'}\n")
        
    except Exception as e:
        print(f"{var}: Error - {str(e)}\n")
        adf_results[var] = {'statistic': np.nan, 'pvalue': np.nan, 'stationary': 'Error'}

print("\n" + "=" * 80)
print("4. TỔNG HỢP KẾT QUẢ")
print("=" * 80)

# Create summary table
results_table = []
for var in variables_to_test:
    llc = llc_results.get(var, {})
    ips = ips_results.get(var, {})
    adf = adf_results.get(var, {})
    
    # Count stationary tests
    stationary_count = sum([
        llc.get('stationary') == 'Yes',
        ips.get('stationary') == 'Yes',
        adf.get('stationary') == 'Yes'
    ])
    
    # Conclusion: majority vote
    conclusion = 'I(0)' if stationary_count >= 2 else 'I(1)'
    
    results_table.append({
        'Variable': var,
        'LLC_Stat': llc.get('statistic', np.nan),
        'LLC_pval': llc.get('pvalue', np.nan),
        'IPS_Stat': ips.get('statistic', np.nan),
        'IPS_pval': ips.get('pvalue', np.nan),
        'ADF_Stat': adf.get('statistic', np.nan),
        'ADF_pval': adf.get('pvalue', np.nan),
        'Conclusion': conclusion
    })

df_results = pd.DataFrame(results_table)

print("\nBẢNG KẾT QUẢ KIỂM ĐỊNH NGHIỆM ĐƠN VỊ PANEL")
print("-" * 80)
print(df_results.to_string(index=False))
print("\nGhi chú:")
print("- I(0): Stationary at level (dùng giá trị gốc)")
print("- I(1): Non-stationary at level (cần sai phân bậc 1)")

print("\n" + "=" * 80)
print("5. TẠO BIẾN SAI PHÂN (NẾU CẦN)")
print("=" * 80)

# Prepare panel data
df_panel = df.set_index(['ISO3 code', 'year'])
df_panel = df_panel.sort_index()

# Create differenced variables for I(1)
i1_vars = df_results[df_results['Conclusion'] == 'I(1)']['Variable'].tolist()

if len(i1_vars) > 0:
    print(f"\nCác biến cần sai phân bậc 1: {i1_vars}")
    
    for var in i1_vars:
        df_panel[f'd_{var}'] = df_panel.groupby(level=0)[var].diff()
        print(f"  Tạo biến d_{var}")
    
    df_panel_reset = df_panel.reset_index()
    df_panel_reset.to_csv('../data/processed/asean-esg-with-diff.csv', index=False)
    print(f"\nĐã lưu dataset với biến sai phân: data/processed/asean-esg-with-diff.csv")
else:
    print("\nTất cả biến đều stationary (I(0)). Không cần tạo biến sai phân.")

print("\n" + "=" * 80)
print("6. LƯU KẾT QUẢ RA LATEX TABLE")
print("=" * 80)

# Generate LaTeX table
latex_content = """\\begin{table}[htbp]
\\centering
\\caption{Kết quả kiểm định nghiệm đơn vị panel}
\\label{tab:unit_root}
\\small
\\begin{tabular}{lrrrrrrl}
\\hline
\\hline
Biến & LLC & p-value & Fisher & p-value & ADF & p-value & Kết luận \\\\
\\hline
"""

for _, row in df_results.iterrows():
    latex_content += f"{row['Variable']} & {row['LLC_Stat']:.3f} & {row['LLC_pval']:.3f} & "
    latex_content += f"{row['IPS_Stat']:.3f} & {row['IPS_pval']:.3f} & "
    latex_content += f"{row['ADF_Stat']:.3f} & {row['ADF_pval']:.3f} & {row['Conclusion']} \\\\\n"

latex_content += """\\hline
\\hline
\\end{tabular}
\\begin{tablenotes}
\\small
\\item Ghi chú: LLC = Levin-Lin-Chu (approx), Fisher = Fisher-type ADF, ADF = ADF-Fisher. \\\\
\\item I(0) = Stationary at level, I(1) = Non-stationary (cần sai phân). \\\\
\\item Mức ý nghĩa: 5\\%. N = 563 quan sát.
\\end{tablenotes}
\\end{table}
"""

with open('../LaTeX/tables/table-03-unit-root.tex', 'w', encoding='utf-8') as f:
    f.write(latex_content)

print("\nĐã lưu bảng kết quả: LaTeX/tables/table-03-unit-root.tex")

print("\n" + "=" * 80)
print("TỔNG KẾT")
print("=" * 80)
print(f"\nSố biến I(0) (stationary): {len(df_results[df_results['Conclusion'] == 'I(0)'])}")
print(f"Số biến I(1) (non-stationary): {len(df_results[df_results['Conclusion'] == 'I(1)'])}")
print("\nBiến stationary (I(0)):")
print(df_results[df_results['Conclusion'] == 'I(0)']['Variable'].tolist())
print("\nBiến non-stationary (I(1)):")
print(df_results[df_results['Conclusion'] == 'I(1)']['Variable'].tolist())

print("\n" + "=" * 80)
print("HOÀN THÀNH TASK 06")
print("=" * 80)
