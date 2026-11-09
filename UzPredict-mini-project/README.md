# UzPredict: Regression Models for Uzbekistan Development Indicators

**AI Mini Project - Project Proposal**
**Team name:** UzPredict
**Option:** B - Data Analysis & Prediction (regression models)

## Team Members

| No. | Full Name | Student ID | Group |
|---|---|---|---|
| 1 | Kim David | [202390211] | E40B |
| 2 | Lim Artyom | [202390259] | E40A |
| 3 | Malyuchenko Bogdan | [202390210] | E40B |
| 4 | Nurjanov Kamal | [202390212] | E40A |

## 1. Project Overview

### 1.1 Problem

Uzbekistan's economy has changed quickly over the past two decades: economic reforms, trade opening, and rapid growth in internet use. Policymakers, students, and investors need a clear, data-based view of which indicators move together with national income, and how well simple models can predict it.

### 1.2 Objectives

1. Collect a yearly dataset for Uzbekistan from public web sources (World Bank Open Data).
2. Explore and clean the data (missing values, time coverage, scaling).
3. Build and compare four regression-based models:
   - **Simple linear regression:** GDP per capita as a function of time.
   - **Polynomial regression (degree 2):** non-linear growth trend of GDP per capita.
   - **Multiple linear regression:** GDP per capita explained by internet use, trade openness, life expectancy, inflation, and unemployment.
   - **Logistic regression:** classify whether a year has high inflation (above 10%) from trade, unemployment, and internet use.
4. Evaluate models on a chronological test set (the most recent years), so the evaluation reflects real forecasting.
5. Present the findings clearly in a short report and slide deck.

### 1.3 Data

| Variable | World Bank code | Role |
|---|---|---|
| GDP per capita (current US$) | NY.GDP.PCAP.CD | Target (regression) |
| Total population | SP.POP.TOTL | Context |
| Inflation, consumer prices (annual %) | FP.CPI.TOTL.ZG | Feature / logistic target |
| Trade (% of GDP) | NE.TRD.GNFS.ZS | Feature |
| Unemployment, total (% of labor force) | SL.UEM.TOTL.ZS | Feature |
| Individuals using the Internet (%) | IT.NET.USER.ZS | Feature |
| Life expectancy at birth (years) | SP.DYN.LE00.IN | Feature |

Source: https://data.worldbank.org/country/uzbekistan (Uzbekistan, annual data).

### 1.4 Method

- Merge indicators by year; keep complete cases for each model.
- Split chronologically: about 80% for training, about 20% (most recent years) for testing.
- Standardize features using training data only, to avoid data leakage.
- Metrics: R², RMSE, and MAE for regression; accuracy, precision, and recall for logistic regression.
- Code: Python standard library only (`src/analysis.py`), so the results are reproducible without installing packages.

### 1.5 Expected Outcomes

- A reproducible dataset (`data/uzbekistan_indicators.csv`) and analysis script.
- A model comparison table (`outputs/results.md`) with test-set metrics.
- Identification of the indicators most associated with GDP per capita and inflation in Uzbekistan.
- A discussion of limitations: small sample size (39 years for GDP per capita, but only 14 years with complete data for the multi-feature models), correlation rather than causation, and data gaps for some indicators.

## 2. Schedule

| Week | Dates (2026) | Work |
|---|---|---|
| 1 | Oct 1 - Oct 7 | Topic selection, data collection, project proposal submitted |
| 2 | Oct 8 - Oct 14 | Data cleaning, exploratory analysis |
| 3 | Oct 15 - Oct 18 | Model implementation and evaluation |
| 4 | Oct 19 | Team presentation in class / lab session |
| 5 | Oct 20 - Oct 21 | Final report (MS Word) and presentation deck (PPT) submitted |

## 3. Repository Structure

```
.
├── README.md                  # This proposal
├── data/
│   ├── raw_*.json             # Raw World Bank API responses
│   └── uzbekistan_indicators.csv   # Merged yearly table (created by the script)
├── src/
│   └── analysis.py            # Data processing and regression models
└── outputs/
    └── results.md             # Model comparison (created by the script)
```

Run the analysis from the project root:

```bash
python3 src/analysis.py
```

## 4. Deliverables Checklist

- [ ] Proposal with team names, student IDs, and group (E40A / E40B)
- [ ] GitHub repository URL sent to the professor
- [ ] In-class presentation (October 19, 2026)
- [ ] Final report (MS Word) and presentation deck (PPT), submitted by October 21, 2026
