# World Bank ESG Data Download Log

**Date:** 2026-03-09
**Task:** Task 01 - Tải dữ liệu World Bank ESG
**Source:** https://esgdata.worldbank.org/data/download

## Files Downloaded

| File | Size | MD5 Checksum | Description |
|------|------|--------------|-------------|
| esgdata_download-2026-01-09.xlsx | 9.9 MB | 1296fe628970c8576e4810517be7d978 | Original XLSX file (4 sheets) |
| esgdata_download-2026-01-09-data.csv | 13 MB | d1a5054c127c5bc29f05004723b29867 | Data sheet extracted to CSV |

## Dataset Statistics

- **Total rows:** 34,208
- **Total columns:** 70
- **Time range:** 1960 - 2025 (66 years)
- **Number of indicators:** 195
- **Number of countries:** 214

## ASEAN Countries Verification

✓ **All 10 ASEAN countries found:**

| ISO3 Code | Country Name |
|-----------|--------------|
| BRN | Brunei Darussalam |
| IDN | Indonesia |
| KHM | Cambodia |
| LAO | Lao PDR |
| MMR | Myanmar |
| MYS | Malaysia |
| PHL | Philippines |
| SGP | Singapore |
| THA | Thailand |
| VNM | Viet Nam |

## Column Structure

**Identifier columns:**
- ISO3 code (country code)
- Economy (country name)
- Indicator code
- Indicator name

**Time series columns:**
- 1960 to 2025 (66 year columns)

## Sample Indicators (First 10)

1. AG.LND.AGRI.ZS - Agricultural land (% of land area)
2. AG.LND.FRST.ZS - Forest area (% of land area)
3. AG.PRD.FOOD.XD - Food production index (2014-2016 = 100)
4. AG.SRF.TOTL.K2 - Surface area (sq. km)
5. CC.EST - Control of Corruption: Estimate
6. EG.CFT.ACCS.ZS - Access to clean fuels and technologies for cooking (% of population)
7. EG.EGY.PRIM.PP.KD - Energy intensity level of primary energy (MJ/$2021 PPP GDP)
8. EG.ELC.ACCS.RU.ZS - Access to electricity, rural (% of rural population)
9. EG.ELC.ACCS.ZS - Access to electricity (% of population)
10. EG.ELC.COAL.ZS - Electricity production from coal sources (% of total)

## XLSX Sheet Structure

1. **Cover** - Introduction and metadata
2. **Framework** - ESG indicator framework description
3. **Metadata** - Country metadata (ISO3 codes, regions, income groups, climate classifications)
4. **Data** - Panel data timeseries values by country/indicator (34,208 rows)

## Notes

- Data downloaded from World Bank Sovereign ESG Data Portal
- Last updated: January 9, 2026
- Includes both core ESG Framework indicators and supplementary datasets
- Country/region classifications follow World Bank Country and Lending Groups (January 2026)
- Climate classifications from Köppen-Geiger climate system

## Next Steps

- Task 02: Làm sạch và ghép dữ liệu ASEAN
- Filter data for 10 ASEAN countries only
- Handle missing values
- Select relevant ESG indicators
