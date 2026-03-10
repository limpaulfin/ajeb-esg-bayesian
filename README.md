# HUB2026-p2-ajeb-green

## Beschreibung

Dieses Repository enthält den Quellcode und die Daten für eine empirische Studie über den Einfluss von ESG-Faktoren (Environmental, Social, Governance) auf das BIP-Wachstum in ASEAN-Ländern. Die Analyse basiert auf Bayes'scher Panel-Regression mit PyMC und umfasst MCMC-Diagnostik, Posterior-Analyse sowie einen Vergleich mit frequentistischen Methoden.

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
- Jupyter Notebook (für `.ipynb`-Dateien)

Erforderliche Pakete:

```
pymc>=5.0  pandas  numpy  matplotlib  seaborn  statsmodels  arviz  scipy
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Verwendung

Die Skripte werden in der folgenden Reihenfolge ausgeführt:

```bash
# 1. Variablenkonstruktion
python code/04-build-variables.py

# 2. Bayes'sche Panel-Regression
python notebooks/bayesian-panel-regression.py

# 3. MCMC-Diagnostik
python notebooks/mcmc-diagnostics.py

# 4. Posterior-Analyse
python notebooks/posterior-analysis.py

# 5. Modellvergleich
python notebooks/model-comparison.py

# 6. Einheitswurzeltests
python notebooks/run_unit_root_tests.py

# 7. Frequentistischer Vergleich (Jupyter Notebook)
jupyter notebook notebooks/frequentist-comparison.ipynb
```

## Datenquellen

Die Rohdaten stammen aus dem [World Bank ESG Data Portal](https://datatopics.worldbank.org/esg/).

Verarbeitete Datensätze befinden sich in `data/processed/`. Das Codebuch (`codebook.md`) dokumentiert alle Variablen.

## Projektstruktur

```
code/               Python-Skripte (Variablenkonstruktion, VIF-Analyse)
notebooks/          Jupyter Notebooks und Python-Skripte (Regression, Diagnostik)
data/raw/           Originaldaten der Weltbank (CSV, XLSX)
data/processed/     Verarbeitete Datensätze und Ergebnisse
LaTeX/              Manuskriptquelldateien
docs/               Technische Berichte
```

## Lizenz

Dieses Projekt steht unter der [MIT-Lizenz](LICENSE).

## Autor

Quellcode vorbereitet von: **Lam Thanh Phong**

## Erklärung zur Forschungsethik

Dieses Repository entspricht den ICMJE- und COPE-Richtlinien für Transparenz und Reproduzierbarkeit in der wissenschaftlichen Forschung.
