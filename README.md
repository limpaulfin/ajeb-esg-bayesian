# Bayes-Panelregression (ASEAN-Daten)

Statistische Analyse mit PyMC 5.x: Bayes'sche Panelregression, MCMC-Diagnostik, Posterior-Analyse und frequentistischer Vergleich.

## Zitierung

```bibtex
@article{lam2026esg,
  title   = {ESG and GDP Growth in ASEAN: A Bayesian Panel Regression Approach},
  author  = {Lam, Hoang Dung and Nguyen, Van Dat and Lam, Thanh Phong
             and Bui, Nguyen Phuong Dung},
  journal = {Asian Journal of Economics and Banking},
  year    = {2026}
}
```

## Voraussetzungen

- Python >= 3.10
- Linux (getestet unter Ubuntu)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verwendung

```bash
# 1. Variablenkonstruktion
python code/04-build-variables.py

# 2. Bayes'sche Panelregression
python notebooks/bayesian-panel-regression.py

# 3. MCMC-Diagnostik
python notebooks/mcmc-diagnostics.py

# 4. Posterior-Analyse
python notebooks/posterior-analysis.py

# 5. Modellvergleich
python notebooks/model-comparison.py

# 6. Einheitswurzeltests
python notebooks/run_unit_root_tests.py

# 7. Frequentistischer Vergleich
jupyter notebook notebooks/frequentist-comparison.ipynb
```

## Projektstruktur

```
code/               Skripte (Variablenkonstruktion, VIF-Analyse)
notebooks/          Regression, Diagnostik, Vergleich
data/raw/           Originaldaten (CSV, XLSX)
data/processed/     Ergebnisse
```

## Datenquelle

[World Bank ESG Data Portal](https://datatopics.worldbank.org/esg/)

## Lizenz

[MIT](LICENSE)

Quellcode vorbereitet von: **Lam Thanh Phong**
