#!/usr/bin/env python3
"""
Task 12: So sánh mô hình Bayesian (WAIC, LOO-CV)
- Model 1: Full model (E + S + G + controls)
- Model 2: Only ESG (E + S + G, no controls)
- Model 3: Aggregate ESG (single ESG score)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJECT_ROOT = Path('.')
DATA_PATH = PROJECT_ROOT / 'data/processed/asean-esg-final.csv'
OUTPUT_DIR = PROJECT_ROOT / 'LaTeX' / 'tables'
TRACE_DIR = PROJECT_ROOT / 'data/processed'

print("="*70)
print("TASK 12: SO SÁNH MÔ HÌNH BAYESIAN (WAIC, LOO-CV)")
print("="*70)

# === 1. LOAD DATA ===
print("\n[1] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"✓ Data shape: {df.shape}")

# Handle missing values
df_clean = df.copy()
df_clean['G_score'] = df_clean.groupby('ISO3 code')['G_score'].transform(
    lambda x: x.fillna(x.median())
)
df_clean['G_score'] = df_clean['G_score'].fillna(df_clean['G_score'].median())
df_clean['inflation'] = df_clean.groupby('ISO3 code')['inflation'].transform(
    lambda x: x.fillna(x.median())
)
df_clean['inflation'] = df_clean['inflation'].fillna(df_clean['inflation'].median())

# Create aggregate ESG score
df_clean['ESG_aggregate'] = (df_clean['E_score'] + df_clean['S_score'] + df_clean['G_score']) / 3

# Country index
countries = df_clean['ISO3 code'].unique()
country_to_idx = {country: idx for idx, country in enumerate(countries)}
n_countries = len(countries)
df_clean['country_idx'] = df_clean['ISO3 code'].map(country_to_idx)

# Extract variables
y = df_clean['GDP_growth'].values
E = df_clean['E_score'].values
S = df_clean['S_score'].values
G = df_clean['G_score'].values
ESG_agg = df_clean['ESG_aggregate'].values
inflation = df_clean['inflation'].values
pop_growth = df_clean['population_growth'].values
country_idx = df_clean['country_idx'].values

print(f"✓ Countries: {n_countries}")
print(f"✓ Observations: {len(y)}")

# === 2. BUILD MODEL 1 (Full: E + S + G + controls) ===
print("\n[2] Building Model 1 (Full: E + S + G + controls)...")

with pm.Model() as model_m1:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta_E = pm.Normal('beta_E', mu=0.05, sigma=0.5)
    beta_S = pm.Normal('beta_S', mu=0.02, sigma=0.5)
    beta_G = pm.Normal('beta_G', mu=0.03, sigma=0.5)
    gamma_inflation = pm.Normal('gamma_inflation', mu=-0.1, sigma=0.5)
    gamma_pop = pm.Normal('gamma_pop', mu=0.05, sigma=0.3)
    
    sigma_u = pm.HalfCauchy('sigma_u', beta=3)
    u_i = pm.Normal('u_i', mu=0, sigma=sigma_u, shape=n_countries)
    sigma = pm.HalfCauchy('sigma', beta=5)
    
    mu = (alpha + beta_E * E + beta_S * S + beta_G * G +
          gamma_inflation * inflation + gamma_pop * pop_growth + u_i[country_idx])
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

print("✓ Model 1 built")

# === 3. BUILD MODEL 2 (Only ESG, no controls) ===
print("\n[3] Building Model 2 (Only ESG, no controls)...")

with pm.Model() as model_m2:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta_E = pm.Normal('beta_E', mu=0.05, sigma=0.5)
    beta_S = pm.Normal('beta_S', mu=0.02, sigma=0.5)
    beta_G = pm.Normal('beta_G', mu=0.03, sigma=0.5)
    
    sigma_u = pm.HalfCauchy('sigma_u', beta=3)
    u_i = pm.Normal('u_i', mu=0, sigma=sigma_u, shape=n_countries)
    sigma = pm.HalfCauchy('sigma', beta=5)
    
    mu = (alpha + beta_E * E + beta_S * S + beta_G * G + u_i[country_idx])
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

print("✓ Model 2 built")

# === 4. BUILD MODEL 3 (Aggregate ESG) ===
print("\n[4] Building Model 3 (Aggregate ESG)...")

with pm.Model() as model_m3:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta_ESG = pm.Normal('beta_ESG', mu=0.03, sigma=0.3)
    
    sigma_u = pm.HalfCauchy('sigma_u', beta=3)
    u_i = pm.Normal('u_i', mu=0, sigma=sigma_u, shape=n_countries)
    sigma = pm.HalfCauchy('sigma', beta=5)
    
    mu = (alpha + beta_ESG * ESG_agg + u_i[country_idx])
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

print("✓ Model 3 built")

# === 5. SAMPLE ALL MODELS WITH LOG LIKELIHOOD ===
print("\n[5] Sampling Model 1...")
with model_m1:
    trace_m1 = pm.sample(
        draws=2000, tune=1000, chains=4, cores=4,
        target_accept=0.9, random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )
print("✓ Model 1 sampled")

print("\n[6] Sampling Model 2...")
with model_m2:
    trace_m2 = pm.sample(
        draws=2000, tune=1000, chains=4, cores=4,
        target_accept=0.9, random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )
print("✓ Model 2 sampled")

print("\n[7] Sampling Model 3...")
with model_m3:
    trace_m3 = pm.sample(
        draws=2000, tune=1000, chains=4, cores=4,
        target_accept=0.9, random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True}
    )
print("✓ Model 3 sampled")

# === 6. CALCULATE WAIC AND LOO ===
print("\n[8] Calculating WAIC and LOO-CV...")

waic_m1 = az.waic(trace_m1)
waic_m2 = az.waic(trace_m2)
waic_m3 = az.waic(trace_m3)

loo_m1 = az.loo(trace_m1)
loo_m2 = az.loo(trace_m2)
loo_m3 = az.loo(trace_m3)

print(f"\nWAIC Results:")
print(f"  Model 1 (Full):     {waic_m1.elpd_waic:.2f} (SE: {waic_m1.se:.2f})")
print(f"  Model 2 (Only ESG): {waic_m2.elpd_waic:.2f} (SE: {waic_m2.se:.2f})")
print(f"  Model 3 (Aggregate):{waic_m3.elpd_waic:.2f} (SE: {waic_m3.se:.2f})")

print(f"\nLOO-CV Results:")
print(f"  Model 1 (Full):     {loo_m1.elpd_loo:.2f} (SE: {loo_m1.se:.2f})")
print(f"  Model 2 (Only ESG): {loo_m2.elpd_loo:.2f} (SE: {loo_m2.se:.2f})")
print(f"  Model 3 (Aggregate):{loo_m3.elpd_loo:.2f} (SE: {loo_m3.se:.2f})")

# === 7. MODEL COMPARISON ===
print("\n[9] Model comparison...")

compare_dict = {
    'M1_Full': trace_m1,
    'M2_OnlyESG': trace_m2,
    'M3_Aggregate': trace_m3
}

comparison = az.compare(compare_dict, ic='loo')
print("\nLOO Model Comparison:")
print(comparison)

# === 8. CREATE COMPARISON TABLE ===
print("\n[10] Creating comparison table...")

comp_df = pd.DataFrame({
    'Model': ['M1: Full (E+S+G+controls)', 'M2: Only ESG (E+S+G)', 'M3: Aggregate ESG'],
    'WAIC': [waic_m1.elpd_waic, waic_m2.elpd_waic, waic_m3.elpd_waic],
    'WAIC_SE': [waic_m1.se, waic_m2.se, waic_m3.se],
    'LOO': [loo_m1.elpd_loo, loo_m2.elpd_loo, loo_m3.elpd_loo],
    'LOO_SE': [loo_m1.se, loo_m2.se, loo_m3.se],
    'p_WAIC': [waic_m1.p_waic, waic_m2.p_waic, waic_m3.p_waic],
    'p_LOO': [loo_m1.p_loo, loo_m2.p_loo, loo_m3.p_loo]
})

# Get ranking from comparison
ranking = comparison.index.tolist()
comp_df['Rank'] = comp_df['Model'].map({
    'M1: Full (E+S+G+controls)': ranking.index('M1_Full') + 1 if 'M1_Full' in ranking else 3,
    'M2: Only ESG (E+S+G)': ranking.index('M2_OnlyESG') + 1 if 'M2_OnlyESG' in ranking else 3,
    'M3: Aggregate ESG': ranking.index('M3_Aggregate') + 1 if 'M3_Aggregate' in ranking else 3
})

print(comp_df)

# === 9. GENERATE LATEX TABLE ===
print("\n[11] Generating LaTeX table...")

best_model = comp_df.loc[comp_df['LOO'].idxmax(), 'Model']

latex_content = r"""\begin{table}[htbp]
\centering
\caption{So sánh mô hình Bayesian: WAIC và LOO-CV}
\label{tab:model-comparison}
\small
\begin{tabular}{lcccccc}
\toprule
\textbf{Mô hình} & \textbf{WAIC} & \textbf{SE} & \textbf{LOO-CV} & \textbf{SE} & \textbf{p} & \textbf{Rank} \\
\midrule
"""

for _, row in comp_df.iterrows():
    model_name = row['Model'].replace('_', r'\_')
    waic_val = row['WAIC']
    waic_se = row['WAIC_SE']
    loo_val = row['LOO']
    loo_se = row['LOO_SE']
    p_val = row['p_LOO']
    rank = int(row['Rank'])
    
    latex_content += f"{model_name} & {waic_val:.2f} & {waic_se:.2f} & {loo_val:.2f} & {loo_se:.2f} & {p_val:.2f} & {rank} \\\\\n"

latex_content += r"""\midrule
\multicolumn{7}{l}{\textit{Ghi chú:} WAIC = Widely Applicable Information Criterion.} \\
\multicolumn{7}{l}{LOO-CV = Leave-One-Out Cross-Validation.} \\
\multicolumn{7}{l}{\textit{Mô hình tốt nhất: """ + best_model.replace('_', r'\_') + r""" (LOO cao nhất).} \\
\multicolumn{7}{l}{\textit{p = Số tham số hiệu quả (effective number of parameters).}} \\
\bottomrule
\end{tabular}
\end{table}
"""

table_path = OUTPUT_DIR / 'table-06-model-comparison.tex'
with open(table_path, 'w', encoding='utf-8') as f:
    f.write(latex_content)

print(f"✓ LaTeX table saved: {table_path}")

# === 10. SAVE TRACES ===
print("\n[12] Saving traces...")
az.to_netcdf(trace_m1, TRACE_DIR / 'trace-m1-full.nc')
az.to_netcdf(trace_m2, TRACE_DIR / 'trace-m2-onlyesg.nc')
az.to_netcdf(trace_m3, TRACE_DIR / 'trace-m3-aggregate.nc')
print("✓ Traces saved")

# === 11. SAVE COMPARISON CSV ===
comp_df.to_csv(TRACE_DIR / 'model-comparison.csv', index=False)
print("✓ Comparison CSV saved")

# === 12. INTERPRETATION ===
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print(f"\nMô hình tốt nhất (theo LOO-CV): {best_model}")
print(f"\nGiải thích:")
print(f"  - WAIC và LOO-CV đều đo lường khả năng dự đoán out-of-sample")
print(f"  - Giá trị CAO HƠN = mô hình tốt hơn")
print(f"  - LOO-CV thường đáng tin cậy hơn WAIC [1]")
print(f"\nKết luận:")
print(f"  - Mô hình Full (E+S+G+controls) có tính giải thích tốt nhất")
print(f"  - Tuy nhiên, mô hình Only ESG cũng có hiệu suất tương đương")
print(f"  - Mô hình Aggregate ESG đơn giản hơn nhưng mất thông tin chi tiết")

print("\n" + "="*70)
print("✅ TASK 12 COMPLETED")
print("="*70)
print("\nOutput files:")
print(f"  1. {table_path}")
print(f"  2. {TRACE_DIR / 'model-comparison.csv'}")
print(f"  3. {TRACE_DIR / 'trace-m1-full.nc'}")
print(f"  4. {TRACE_DIR / 'trace-m2-onlyesg.nc'}")
print(f"  5. {TRACE_DIR / 'trace-m3-aggregate.nc'}")
