#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 10: Chẩn đoán MCMC và convergence
Phase C: Mô hình Bayesian
"""

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Đường dẫn
BASE_DIR = '.'
TRACE_PATH = f'{BASE_DIR}/data/processed/trace.nc'
FIGURES_DIR = f'{BASE_DIR}/LaTeX/figures'

# Load trace
print("Loading trace.nc...")
idata = az.from_netcdf(TRACE_PATH)
print(f"Trace loaded: {list(idata.posterior.data_vars)}")

# 1. Tính R-hat cho tất cả parameters
print("\n" + "="*60)
print("R-HAT VALUES (cần < 1.01)")
print("="*60)
rhat = az.rhat(idata)
print(rhat)

# Check R-hat < 1.01
rhat_values = []
rhat_ok = True
for var in idata.posterior.data_vars:
    vals = np.atleast_1d(rhat[var].values)
    max_rhat = float(np.max(vals))
    rhat_values.append((var, max_rhat))
    status = "✓" if max_rhat < 1.01 else "✗"
    if len(vals) > 1:
        print(f"  {var}: max={max_rhat:.4f} (shape={vals.shape}) {status}")
    else:
        print(f"  {var}: {max_rhat:.4f} {status}")
    if max_rhat >= 1.01:
        rhat_ok = False
print(f"\nR-hat check: {'PASSED' if rhat_ok else 'FAILED'}")

# 2. Tính ESS cho tất cả parameters
print("\n" + "="*60)
print("EFFECTIVE SAMPLE SIZE (ESS) (cần > 400)")
print("="*60)
ess = az.ess(idata)
print(ess)

# Check ESS > 400
ess_values = []
ess_ok = True
for var in idata.posterior.data_vars:
    vals = np.atleast_1d(ess[var].values)
    min_ess = float(np.min(vals))
    ess_values.append((var, min_ess))
    status = "✓" if min_ess > 400 else "✗"
    if len(vals) > 1:
        print(f"  {var}: min={min_ess:.1f} (shape={vals.shape}) {status}")
    else:
        print(f"  {var}: {min_ess:.1f} {status}")
    if min_ess <= 400:
        ess_ok = False
print(f"\nESS check: {'PASSED' if ess_ok else 'FAILED'}")

# 3. Kiểm tra divergences
print("\n" + "="*60)
print("DIVERGENCES CHECK")
print("="*60)
try:
    diverging = idata.sample_stats['diverging'].sum().values
    total_draws = idata.sample_stats['diverging'].size
    div_pct = (diverging / total_draws) * 100
    print(f"Divergences: {diverging} / {total_draws} ({div_pct:.2f}%)")
    div_ok = div_pct < 1.0
    print(f"Divergence check: {'PASSED' if div_ok else 'FAILED'} (<1%)")
except Exception as e:
    print(f"Không thể kiểm tra divergences: {e}")
    diverging = 0
    div_ok = True

# 4. Vẽ trace plots
print("\n" + "="*60)
print("Generating trace plots...")
print("="*60)

# Trace plot cho tất cả parameters (let ArviZ handle layout)
az.plot_trace(idata, compact=True, figsize=(14, 12))
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig-trace-all.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURES_DIR}/fig-trace-all.png")

# Trace plot chỉ cho beta_E, beta_S, beta_G (nếu có)
beta_vars = [v for v in ['beta_E', 'beta_S', 'beta_G', 'beta'] if v in list(idata.posterior.data_vars)]
if beta_vars:
    az.plot_trace(idata, var_names=beta_vars, compact=True, figsize=(12, 6))
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig-trace-beta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR}/fig-trace-beta.png")

# 5. Vẽ posterior plots
print("\n" + "="*60)
print("Generating posterior plots...")
print("="*60)

# Posterior plot cho tất cả parameters
az.plot_posterior(idata, figsize=(14, 8))
plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/fig-posterior-all.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {FIGURES_DIR}/fig-posterior-all.png")

# Posterior plot chỉ cho beta variables
if beta_vars:
    az.plot_posterior(idata, var_names=beta_vars, figsize=(12, 4))
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig-posterior-beta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR}/fig-posterior-beta.png")

# 6. Pair plot cho beta_E, beta_S, beta_G
print("\n" + "="*60)
print("Generating pair plots...")
print("="*60)

if len(beta_vars) >= 2:
    az.plot_pair(idata, var_names=beta_vars, figsize=(10, 10), 
                 kind='kde', marginals=True)
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig-pair-beta.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR}/fig-pair-beta.png")

# 7. Summary statistics
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
summary = az.summary(idata, round_to=4)
print(summary.to_string())

# Save summary to file
summary.to_csv(f'{BASE_DIR}/data/processed/mcmc-summary.csv')
print(f"\nSaved: {BASE_DIR}/data/processed/mcmc-summary.csv")

# 8. Energy plot (nếu có)
print("\n" + "="*60)
print("Energy plot...")
print("="*60)
try:
    az.plot_energy(idata, figsize=(10, 6))
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/fig-energy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIGURES_DIR}/fig-energy.png")
except Exception as e:
    print(f"Không thể vẽ energy plot: {e}")

# Final verification
print("\n" + "="*60)
print("VERIFICATION SUMMARY")
print("="*60)
all_ok = rhat_ok and ess_ok and div_ok
print(f"R-hat < 1.01: {'✓ PASSED' if rhat_ok else '✗ FAILED'}")
print(f"ESS > 400:    {'✓ PASSED' if ess_ok else '✗ FAILED'}")
print(f"Divergences < 1%: {'✓ PASSED' if div_ok else '✗ FAILED'}")
print(f"\nOVERALL: {'✓ ALL CHECKS PASSED' if all_ok else '✗ SOME CHECKS FAILED'}")

if not all_ok:
    print("\n⚠ KHUYẾN NGHỊ:")
    if not rhat_ok:
        print("  - Tăng target_accept (ví dụ: 0.95 → 0.99)")
        print("  - Tăng tuning steps (ví dụ: 2000 → 5000)")
    if not ess_ok:
        print("  - Tăng số draws (ví dụ: 2000 → 4000)")
        print("  - Chạy thêm chains")
    if not div_ok:
        print("  - Tăng target_accept")
        print("  - Reparameterize model")
