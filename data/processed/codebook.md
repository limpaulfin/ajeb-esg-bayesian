# Codebook: ASEAN ESG Dataset

**Generated:** 2026-03-09  
**Source:** World Bank Sovereign ESG Data Portal  
**Countries:** 10 ASEAN countries  
**Period:** 1961 - 2024  
**Observations:** 563

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
| `GDP_growth` | Tốc độ tăng trưởng GDP hàng năm (%) | Numeric | [-34.81, 24.34] | NY.GDP.MKTP.KD.ZG |

### Independent Variables (IV) - ESG Scores

**Methodology:** Composite scores calculated using min-max normalization (0-100 scale) across 87 Environmental, 28 Social, and 6 Governance indicators.

**References:**
- World Bank Sovereign ESG Data Portal (2023)
- Thomson Reuters ESG Scoring Methodology (2017)

| Variable | Description | Type | Mean (SD) | Indicators Used |
|----------|-------------|------|-----------|-----------------|
| `E_score` | Điểm Environmental (0-100) | Numeric | 29.54 (8.02) | 87 indicators |
| `S_score` | Điểm Social (0-100) | Numeric | 47.97 (5.82) | 28 indicators |
| `G_score` | Điểm Governance (0-100) | Numeric | 21.86 (23.32) | 6 indicators |

#### Environmental Indicators (87 total)
- `AG.LND.AGRI.ZS`
- `AG.LND.FRST.ZS`
- `AG.PRD.FOOD.XD`
- `AG.SRF.TOTL.K2`
- `EG.CFT.ACCS.ZS`
- `EG.EGY.PRIM.PP.KD`
- `EG.ELC.ACCS.RU.ZS`
- `EG.ELC.ACCS.ZS`
- `EG.ELC.COAL.ZS`
- `EG.ELC.RNEW.ZS`
... (see esg-indicator-classification.csv for full list)

#### Social Indicators (28 total)
- `HD.HCI.OVRL`
- `SE.ADT.LITR.ZS`
- `SE.PRM.CMPT.ZS`
- `SE.XPD.TOTL.GB.ZS`
- `SH.DTH.COMM.ZS`
- `SH.DYN.MORT`
- `SH.H2O.SMDW.ZS`
- `SH.MED.BEDS.ZS`
- `SH.STA.SMSS.ZS`
- `SI.SPR.PGAP`
... (see esg-indicator-classification.csv for full list)

#### Governance Indicators (6 total)
- `GB.XPD.RSDV.GD.ZS`
- `SI.POV.UMIC`
- `NW.NCA.MNIC.PC`
- `NW.NCA.MNIC.TO`
- `NW.NCA.MSIL.PC`
- `NW.NCA.MSIL.TO`
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
| `inflation` | Lạm phát giá tiêu dùng hàng năm (%) | Numeric | 10.76 (53.69) | FP.CPI.TOTL.ZG |
| `population_growth` | Tăng trưởng dân số hàng năm (%) | Numeric | 1.90 (1.27) | SP.POP.GROW |

**Note:** Control variables FDI và trade openness chưa có trong World Bank Sovereign ESG dataset. Cần bổ sung từ World Bank WDI (World Development Indicators):
- `FDI_gdp`: BX.KLT.DINV.WD.GD.ZS (FDI net inflows % of GDP)
- `trade_openness`: (NE.EXP.GNFS.ZS + NE.IMP.GNFS.ZS) / 2

---

## Data Quality

### Missing Values

| Variable | Missing Count | Missing % |
|----------|---------------|-----------|
| GDP_growth | 0 | 0.0% |
| E_score | 0 | 0.0% |
| S_score | 0 | 0.0% |
| G_score | 306 | 54.4% |
| inflation | 46 | 8.2% |
| population_growth | 0 | 0.0% |

### Correlation Matrix

```
                   GDP_growth  E_score  S_score  G_score  inflation  population_growth
GDP_growth              1.000    0.029   -0.040   -0.000     -0.045              0.072
E_score                 0.029    1.000    0.161    0.270      0.004             -0.324
S_score                -0.040    0.161    1.000   -0.072     -0.054              0.345
G_score                -0.000    0.270   -0.072    1.000      0.181             -0.008
inflation              -0.045    0.004   -0.054    0.181      1.000              0.052
population_growth       0.072   -0.324    0.345   -0.008      0.052              1.000
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
