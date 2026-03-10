#!/usr/bin/env python3
"""
Task 04: Xây dựng biến và hoàn thiện dataset
Date: 2026-03-09

Purpose:
- Tạo biến phụ thuộc (DV): GDP growth rate
- Tạo biến độc lập (IV): E_score, S_score, G_score
- Tạo biến kiểm soát (control): inflation, population_growth
- Tạo biến lag nếu cần
- Lưu dataset cuối cùng: asean-esg-final.csv
- Tạo codebook: codebook.md

References:
- World Bank Sovereign ESG Data Portal (2023)
- Thomson Reuters ESG Scoring Methodology (2017)
- Blanchard, O. (2021). Macroeconomics (8th ed.). Pearson.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data' / 'processed'
OUTPUT_DIR = DATA_DIR

print("=" * 80)
print("TASK 04: Xây dựng biến và hoàn thiện dataset")
print("=" * 80)

# Load data
print("\n[1/7] Đọc dữ liệu...")
df = pd.read_csv(DATA_DIR / 'asean-esg-panel.csv')
print(f"  Dataset shape: {df.shape}")
print(f"  Countries: {df['ISO3 code'].nunique()}")
print(f"  Years: {df['year'].min()} - {df['year'].max()}")

# Load indicator classification
print("\n[2/7] Đọc phân loại chỉ số ESG...")
indicator_class = pd.read_csv(DATA_DIR / 'esg-indicator-classification.csv')
print(f"  Total indicators: {len(indicator_class)}")
print(f"  E indicators: {len(indicator_class[indicator_class['Pillar'] == 'E'])}")
print(f"  S indicators: {len(indicator_class[indicator_class['Pillar'] == 'S'])}")
print(f"  G indicators: {len(indicator_class[indicator_class['Pillar'] == 'G'])}")
print(f"  Unknown: {len(indicator_class[indicator_class['Pillar'] == 'Unknown'])}")

# Define indicator groups
print("\n[3/7] Phân nhóm chỉ số E, S, G...")

# Get indicators for each pillar (only those available in dataset)
e_indicators = indicator_class[
    (indicator_class['Pillar'] == 'E') & 
    (indicator_class['Indicator code'].isin(df.columns))
]['Indicator code'].tolist()

s_indicators = indicator_class[
    (indicator_class['Pillar'] == 'S') & 
    (indicator_class['Indicator code'].isin(df.columns))
]['Indicator code'].tolist()

g_indicators = indicator_class[
    (indicator_class['Pillar'] == 'G') & 
    (indicator_class['Indicator code'].isin(df.columns))
]['Indicator code'].tolist()

print(f"  E indicators available: {len(e_indicators)}")
print(f"  S indicators available: {len(s_indicators)}")
print(f"  G indicators available: {len(g_indicators)}")

# Function to normalize indicators to 0-100 scale
def normalize_to_100(series, reverse=False):
    """
    Normalize series to 0-100 scale using min-max normalization.
    
    Args:
        series: pd.Series to normalize
        reverse: if True, reverse scale (higher = worse, e.g., emissions)
    
    Returns:
        Normalized series (0-100 scale)
    """
    # Remove NaN for calculation
    valid_data = series.dropna()
    
    if len(valid_data) == 0:
        return series
    
    min_val = valid_data.min()
    max_val = valid_data.max()
    
    if min_val == max_val:
        # All values are the same
        return pd.Series([50.0] * len(series), index=series.index)
    
    # Min-max normalization to 0-100
    normalized = ((series - min_val) / (max_val - min_val)) * 100
    
    # Reverse if needed (for negative indicators like emissions)
    if reverse:
        normalized = 100 - normalized
    
    return normalized

# Determine reverse indicators (higher = worse)
reverse_indicators = [
    # Emissions (higher = worse)
    'EN.GHG.ALL.MT.CE.AR5', 'EN.GHG.ALL.PC.CE.AR5',
    'EN.GHG.CO2.MT.CE.AR5', 'EN.GHG.CO2.PC.CE.AR5',
    'EN.ATM.PM25.MC.M3',
    # Resource depletion (higher = worse)
    'NY.ADJ.DFOR.GN.ZS', 'NY.ADJ.DRES.GN.ZS',
    'ER.H2O.FWST.ZS', 'ER.H2O.FWTL.ZS',
    # Unemployment (higher = worse)
    'SL.UEM.TOTL.ZS',
    # Poverty (higher = worse)
    'SI.POV.DDAY', 'SI.POV.UMIC', 'SI.SPR.PGAP',
    # Mortality (higher = worse)
    'SH.DYN.MORT',
    # Disease burden (higher = worse)
    'SH.DTH.COMM.ZS',
]

# Calculate E, S, G scores
print("\n[4/7] Tính điểm E, S, G (chuẩn hóa 0-100)...")

# Initialize score columns
df['E_score'] = np.nan
df['S_score'] = np.nan
df['G_score'] = np.nan

# Calculate E score
print("  Calculating E_score...")
e_scores = []
for indicator in e_indicators:
    if indicator in df.columns:
        reverse = indicator in reverse_indicators
        normalized = normalize_to_100(df[indicator], reverse=reverse)
        e_scores.append(normalized)

if e_scores:
    e_score_matrix = pd.concat(e_scores, axis=1)
    df['E_score'] = e_score_matrix.mean(axis=1, skipna=True)

# Calculate S score
print("  Calculating S_score...")
s_scores = []
for indicator in s_indicators:
    if indicator in df.columns:
        reverse = indicator in reverse_indicators
        normalized = normalize_to_100(df[indicator], reverse=reverse)
        s_scores.append(normalized)

if s_scores:
    s_score_matrix = pd.concat(s_scores, axis=1)
    df['S_score'] = s_score_matrix.mean(axis=1, skipna=True)

# Calculate G score
print("  Calculating G_score...")
g_scores = []
for indicator in g_indicators:
    if indicator in df.columns:
        # Governance indicators typically: higher = better
        normalized = normalize_to_100(df[indicator], reverse=False)
        g_scores.append(normalized)

if g_scores:
    g_score_matrix = pd.concat(g_scores, axis=1)
    df['G_score'] = g_score_matrix.mean(axis=1, skipna=True)

print(f"  E_score: mean={df['E_score'].mean():.2f}, std={df['E_score'].std():.2f}")
print(f"  S_score: mean={df['S_score'].mean():.2f}, std={df['S_score'].std():.2f}")
print(f"  G_score: mean={df['G_score'].mean():.2f}, std={df['G_score'].std():.2f}")

# Create dependent variable (DV)
print("\n[5/7] Tạo biến phụ thuộc và kiểm soát...")
df['GDP_growth'] = df['NY.GDP.MKTP.KD.ZG']  # GDP growth (annual %)
print(f"  GDP_growth: mean={df['GDP_growth'].mean():.2f}%, std={df['GDP_growth'].std():.2f}")

# Create control variables
df['inflation'] = df['FP.CPI.TOTL.ZG']  # Inflation (annual %)
df['population_growth'] = df['SP.POP.GROW']  # Population growth (annual %)

print(f"  inflation: mean={df['inflation'].mean():.2f}%, std={df['inflation'].std():.2f}")
print(f"  population_growth: mean={df['population_growth'].mean():.2f}%, std={df['population_growth'].std():.2f}")

# Create lag variables (1-year lag for E, S, G scores)
print("\n[6/7] Tạo biến lag (1 năm)...")
df = df.sort_values(['ISO3 code', 'year'])

for var in ['E_score', 'S_score', 'G_score']:
    df[f'{var}_lag1'] = df.groupby('ISO3 code')[var].shift(1)
    print(f"  {var}_lag1 created")

# Select final columns for dataset
final_columns = [
    # Identifiers
    'ISO3 code', 'Economy', 'year',
    # Dependent variable
    'GDP_growth',
    # Independent variables
    'E_score', 'S_score', 'G_score',
    # Lag variables
    'E_score_lag1', 'S_score_lag1', 'G_score_lag1',
    # Control variables
    'inflation', 'population_growth',
]

# Create final dataset
df_final = df[final_columns].copy()

# Remove rows with missing DV
initial_rows = len(df_final)
df_final = df_final.dropna(subset=['GDP_growth'])
final_rows = len(df_final)
print(f"\n[7/7] Loại bỏ missing DV: {initial_rows} → {final_rows} rows ({initial_rows - final_rows} removed)")

# Save final dataset
output_file = OUTPUT_DIR / 'asean-esg-final.csv'
df_final.to_csv(output_file, index=False)
print(f"\n✓ Dataset saved: {output_file}")
print(f"  Shape: {df_final.shape}")
print(f"  Columns: {len(df_final.columns)}")

# Summary statistics
print("\n" + "=" * 80)
print("THỐNG KÊ MÔ TẢ - DATASET CUỐI CÙNG")
print("=" * 80)
print(df_final.describe().round(2).to_string())

# Correlation matrix
print("\n" + "=" * 80)
print("MA TRẬN TƯƠNG QUAN")
print("=" * 80)
corr_vars = ['GDP_growth', 'E_score', 'S_score', 'G_score', 'inflation', 'population_growth']
corr_matrix = df_final[corr_vars].corr().round(3)
print(corr_matrix.to_string())

# Create codebook
print("\n" + "=" * 80)
print("TẠO CODEBOOK")
print("=" * 80)

codebook_content = f"""# Codebook: ASEAN ESG Dataset

**Generated:** 2026-03-09  
**Source:** World Bank Sovereign ESG Data Portal  
**Countries:** {df_final['ISO3 code'].nunique()} ASEAN countries  
**Period:** {df_final['year'].min()} - {df_final['year'].max()}  
**Observations:** {len(df_final)}

---

## Variables

### Identifiers

| Variable | Description | Type |
|----------|-------------|------|
| `ISO3 code` | Mã quốc gia ISO 3-letter | String |
| `Economy` | Tên quốc gia | String |
| `year` | Năm | Integer |

### Dependent Variable (DV)

| Variable | Description | Type | Range | Source |
|----------|-------------|------|-------|--------|
| `GDP_growth` | Tốc độ tăng trưởng GDP hàng năm (%) | Numeric | [{df_final['GDP_growth'].min():.2f}, {df_final['GDP_growth'].max():.2f}] | NY.GDP.MKTP.KD.ZG |

### Independent Variables (IV) - ESG Scores

**Methodology:** Composite scores calculated using min-max normalization (0-100 scale) across {len(e_indicators)} Environmental, {len(s_indicators)} Social, and {len(g_indicators)} Governance indicators.

**References:**
- World Bank Sovereign ESG Data Portal (2023)
- Thomson Reuters ESG Scoring Methodology (2017)

| Variable | Description | Type | Mean (SD) | Indicators Used |
|----------|-------------|------|-----------|-----------------|
| `E_score` | Điểm Environmental (0-100) | Numeric | {df_final['E_score'].mean():.2f} ({df_final['E_score'].std():.2f}) | {len(e_indicators)} indicators |
| `S_score` | Điểm Social (0-100) | Numeric | {df_final['S_score'].mean():.2f} ({df_final['S_score'].std():.2f}) | {len(s_indicators)} indicators |
| `G_score` | Điểm Governance (0-100) | Numeric | {df_final['G_score'].mean():.2f} ({df_final['G_score'].std():.2f}) | {len(g_indicators)} indicators |

#### Environmental Indicators ({len(e_indicators)} total)
{chr(10).join([f"- `{ind}`" for ind in e_indicators[:10]])}
... (see esg-indicator-classification.csv for full list)

#### Social Indicators ({len(s_indicators)} total)
{chr(10).join([f"- `{ind}`" for ind in s_indicators[:10]])}
... (see esg-indicator-classification.csv for full list)

#### Governance Indicators ({len(g_indicators)} total)
{chr(10).join([f"- `{ind}`" for ind in g_indicators[:10]])}
... (see esg-indicator-classification.csv for full list)

### Lag Variables

| Variable | Description | Type | Purpose |
|----------|-------------|------|---------|
| `E_score_lag1` | E_score trễ 1 năm | Numeric | Mitigate endogeneity |
| `S_score_lag1` | S_score trễ 1 năm | Numeric | Mitigate endogeneity |
| `G_score_lag1` | G_score trễ 1 năm | Numeric | Mitigate endogeneity |

### Control Variables

| Variable | Description | Type | Mean (SD) | Source |
|----------|-------------|------|-----------|--------|
| `inflation` | Lạm phát giá tiêu dùng hàng năm (%) | Numeric | {df_final['inflation'].mean():.2f} ({df_final['inflation'].std():.2f}) | FP.CPI.TOTL.ZG |
| `population_growth` | Tăng trưởng dân số hàng năm (%) | Numeric | {df_final['population_growth'].mean():.2f} ({df_final['population_growth'].std():.2f}) | SP.POP.GROW |

**Note:** Control variables FDI và trade openness chưa có trong World Bank Sovereign ESG dataset. Cần bổ sung từ World Bank WDI (World Development Indicators):
- `FDI_gdp`: BX.KLT.DINV.WD.GD.ZS (FDI net inflows % of GDP)
- `trade_openness`: (NE.EXP.GNFS.ZS + NE.IMP.GNFS.ZS) / 2

---

## Data Quality

### Missing Values

| Variable | Missing Count | Missing % |
|----------|---------------|-----------|
| GDP_growth | {df_final['GDP_growth'].isna().sum()} | {df_final['GDP_growth'].isna().sum() / len(df_final) * 100:.1f}% |
| E_score | {df_final['E_score'].isna().sum()} | {df_final['E_score'].isna().sum() / len(df_final) * 100:.1f}% |
| S_score | {df_final['S_score'].isna().sum()} | {df_final['S_score'].isna().sum() / len(df_final) * 100:.1f}% |
| G_score | {df_final['G_score'].isna().sum()} | {df_final['G_score'].isna().sum() / len(df_final) * 100:.1f}% |
| inflation | {df_final['inflation'].isna().sum()} | {df_final['inflation'].isna().sum() / len(df_final) * 100:.1f}% |
| population_growth | {df_final['population_growth'].isna().sum()} | {df_final['population_growth'].isna().sum() / len(df_final) * 100:.1f}% |

### Correlation Matrix

```
{corr_matrix.to_string()}
```

---

## Model Specification

**Bayesian Panel Regression Model:**

```
GDP_growth_it = α + β1·E_score_it + β2·S_score_it + β3·G_score_it 
               + γ1·inflation_it + γ2·population_growth_it + μ_i + ε_it
```

Where:
- `i` = country, `t` = year
- `μ_i` = country fixed effects
- `ε_it` = error term

**Priors (to be specified in Task 08):**
- α ~ Normal(0, 10)
- β_k ~ Normal(0, 5)
- γ_k ~ Normal(0, 5)
- σ ~ HalfNormal(5)

---

## Next Steps

1. **Task 05:** Ma trận tương quan và đa cộng tuyến
2. **Task 06:** Kiểm định nghiệm đơn vị panel
3. **Task 08:** Xác định prior từ nghiên cứu trước
4. **Data Enhancement:** Bổ sung FDI và trade openness từ WDI

---

## References

1. World Bank Sovereign ESG Data Portal (2023). https://esgdata.worldbank.org
2. Blanchard, O. (2021). Macroeconomics (8th ed.). Pearson. p. 375
3. Thomson Reuters ESG Scores Methodology (2017)
4. Certificate in ESG Investing (2023-2024). p. 425

---

**Date:** 2026-03-09
"""

# Save codebook
codebook_file = OUTPUT_DIR / 'codebook.md'
with open(codebook_file, 'w', encoding='utf-8') as f:
    f.write(codebook_content)

print(f"✓ Codebook saved: {codebook_file}")

# Summary
print("\n" + "=" * 80)
print("HOÀN THÀNH TASK 04")
print("=" * 80)
print(f"✓ Dataset: {output_file}")
print(f"✓ Codebook: {codebook_file}")
print(f"✓ Observations: {len(df_final)}")
print(f"✓ Countries: {df_final['ISO3 code'].nunique()}")
print(f"✓ Variables: {len(df_final.columns)}")
print(f"✓ E indicators: {len(e_indicators)}")
print(f"✓ S indicators: {len(s_indicators)}")
print(f"✓ G indicators: {len(g_indicators)}")
print("\n⚠️  Note: FDI và trade openness cần bổ sung từ World Bank WDI")
print("=" * 80)
