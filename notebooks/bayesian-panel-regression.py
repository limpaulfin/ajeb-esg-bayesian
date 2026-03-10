import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

print(f"PyMC version: {pm.__version__}")
print(f"ArviZ version: {az.__version__}")

# === CELL ===

# Load data
df = pd.read_csv('./data/processed/asean-esg-final.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nCountries: {df['ISO3 code'].nunique()}")
print(f"Years: {df['year'].min()} - {df['year'].max()}")
print(f"\nMissing values:")
print(df[['GDP_growth', 'E_score', 'S_score', 'G_score', 'inflation', 'population_growth']].isnull().sum())

# === CELL ===

# Xử lý missing values
# G_score có nhiều missing values, ta sẽ điền bằng median theo quốc gia
# Inflation điền bằng median theo quốc gia

df_clean = df.copy()

# Điền G_score bằng median theo quốc gia
df_clean['G_score'] = df_clean.groupby('ISO3 code')['G_score'].transform(
    lambda x: x.fillna(x.median())
)

# Nếu vẫn còn missing (do cả quốc gia không có data), điền bằng overall median
df_clean['G_score'] = df_clean['G_score'].fillna(df_clean['G_score'].median())

# Điền inflation bằng median theo quốc gia
df_clean['inflation'] = df_clean.groupby('ISO3 code')['inflation'].transform(
    lambda x: x.fillna(x.median())
)

# Nếu vẫn còn missing, điền bằng overall median
df_clean['inflation'] = df_clean['inflation'].fillna(df_clean['inflation'].median())

print("After imputation:")
print(df_clean[['GDP_growth', 'E_score', 'S_score', 'G_score', 'inflation', 'population_growth']].isnull().sum())

# === CELL ===

# Tạo country index cho random effects
countries = df_clean['ISO3 code'].unique()
country_to_idx = {country: idx for idx, country in enumerate(countries)}
n_countries = len(countries)

df_clean['country_idx'] = df_clean['ISO3 code'].map(country_to_idx)

print(f"Number of countries: {n_countries}")
print(f"\nCountry mapping:")
for country, idx in sorted(country_to_idx.items(), key=lambda x: x[1]):
    print(f"  {idx}: {country}")

# === CELL ===

# Extract variables
y = df_clean['GDP_growth'].values
E = df_clean['E_score'].values
S = df_clean['S_score'].values
G = df_clean['G_score'].values
inflation = df_clean['inflation'].values
pop_growth = df_clean['population_growth'].values
country_idx = df_clean['country_idx'].values

print(f"Data shapes:")
print(f"  y: {y.shape}")
print(f"  E: {E.shape}")
print(f"  S: {S.shape}")
print(f"  G: {G.shape}")
print(f"  inflation: {inflation.shape}")
print(f"  pop_growth: {pop_growth.shape}")
print(f"  country_idx: {country_idx.shape}")

# === CELL ===

# Build PyMC model
with pm.Model() as bayesian_panel_model:
    
    # === Priors ===
    
    # Intercept
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    
    # ESG coefficients
    beta_E = pm.Normal('beta_E', mu=0.05, sigma=0.5)
    beta_S = pm.Normal('beta_S', mu=0.02, sigma=0.5)
    beta_G = pm.Normal('beta_G', mu=0.03, sigma=0.5)
    
    # Control variables
    gamma_inflation = pm.Normal('gamma_inflation', mu=-0.1, sigma=0.5)
    gamma_pop = pm.Normal('gamma_pop', mu=0.05, sigma=0.3)
    
    # Random effects (country-specific)
    sigma_u = pm.HalfCauchy('sigma_u', beta=3)
    u_i = pm.Normal('u_i', mu=0, sigma=sigma_u, shape=n_countries)
    
    # Residual error
    sigma = pm.HalfCauchy('sigma', beta=5)
    
    # === Linear Model ===
    
    mu = (alpha + 
          beta_E * E + 
          beta_S * S + 
          beta_G * G +
          gamma_inflation * inflation +
          gamma_pop * pop_growth +
          u_i[country_idx])
    
    # === Likelihood ===
    
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

print("Model compiled successfully!")
print(bayesian_panel_model)

# === CELL ===

# Run MCMC sampling
# Parameters: 2000 draws per chain, 4 chains, target_accept=0.9

with bayesian_panel_model:
    trace = pm.sample(
        draws=2000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.9,
        random_seed=42,
        return_inferencedata=True
    )

print("\nMCMC sampling completed!")
print(f"Trace type: {type(trace)}")

# === CELL ===

# Save trace to NetCDF file
trace_path = './data/processed/trace.nc'
az.to_netcdf(trace, trace_path)

print(f"Trace saved to: {trace_path}")
import os
file_size_mb = os.path.getsize(trace_path) / 1024 / 1024
print(f"File size: {file_size_mb:.2f} MB")

# === CELL ===

# Summary statistics
az.summary(trace, var_names=['alpha', 'beta_E', 'beta_S', 'beta_G', 
                              'gamma_inflation', 'gamma_pop', 'sigma', 'sigma_u'])

# === CELL ===

# Check convergence diagnostics
print("Convergence Diagnostics:")
print(f"\nNumber of divergences: {trace.sample_stats.diverging.sum().values}")
print(f"\nR-hat statistics (should be ~1.0):")
rhat = az.rhat(trace, var_names=['alpha', 'beta_E', 'beta_S', 'beta_G', 
                                   'gamma_inflation', 'gamma_pop', 'sigma', 'sigma_u'])
print(rhat)

# === CELL ===

# Plot trace
az.plot_trace(trace, var_names=['alpha', 'beta_E', 'beta_S', 'beta_G', 
                                 'gamma_inflation', 'gamma_pop', 'sigma', 'sigma_u'],
              figsize=(15, 20))
plt.tight_layout()
plt.savefig('./output/trace-plot.png', dpi=150, bbox_inches='tight')
plt.show()

# === CELL ===

# Posterior means and 95% credible intervals
print("Posterior Estimates (Mean ± 95% CI):")
print("="*70)

params = ['alpha', 'beta_E', 'beta_S', 'beta_G', 'gamma_inflation', 'gamma_pop', 'sigma', 'sigma_u']

for param in params:
    mean_val = trace.posterior[param].mean().values
    hdi_low = az.hdi(trace.posterior[param], hdi_prob=0.95).sel(hdi='lower').values
    hdi_high = az.hdi(trace.posterior[param], hdi_prob=0.95).sel(hdi='higher').values
    
    print(f"{param:20s}: {mean_val:8.4f} [{hdi_low:7.4f}, {hdi_high:7.4f}]")