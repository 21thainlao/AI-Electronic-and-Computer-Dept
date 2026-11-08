#!/usr/bin/env python3
"""UzPredict - regression analysis of Uzbekistan development indicators.

Data: World Bank Open Data (raw JSON files in data/).
Uses only the Python standard library, so no installation is needed.

Run from the project root:
    python3 src/analysis.py

Outputs:
    data/uzbekistan_indicators.csv  - merged yearly table
    outputs/results.md              - model comparison and coefficients
"""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# Short name -> World Bank indicator code
INDICATORS = {
    "gdp_pc": "NY.GDP.PCAP.CD",       # GDP per capita (current US$) - target
    "population": "SP.POP.TOTL",      # Total population
    "inflation": "FP.CPI.TOTL.ZG",    # Inflation, consumer prices (annual %)
    "trade": "NE.TRD.GNFS.ZS",        # Trade (% of GDP)
    "unemployment": "SL.UEM.TOTL.ZS", # Unemployment, total (% of labor force)
    "internet": "IT.NET.USER.ZS",     # Individuals using the Internet (%)
    "life_exp": "SP.DYN.LE00.IN",     # Life expectancy at birth (years)
}


# ---------------------------------------------------------------- data loading
def load_series(code):
    """Return {year: value} for one World Bank indicator (missing values dropped)."""
    with open(DATA / f"raw_{code}.json", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload[1] or []
    return {int(r["date"]): float(r["value"]) for r in rows if r["value"] is not None}


def save_merged_csv(series):
    years = sorted(set().union(*[set(s) for s in series.values()]))
    path = DATA / "uzbekistan_indicators.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["year"] + list(series))
        for y in years:
            writer.writerow([y] + [series[k].get(y, "") for k in series])
    print(f"Saved merged table: {path}")


def make_table(series, features, target):
    """Complete-case rows (years where target and all features exist)."""
    years = sorted(y for y in series[target] if all(y in series[f] for f in features))
    X = [[series[f][y] for f in features] for y in years]
    y_vals = [series[target][y] for y in years]
    return years, X, y_vals


# ---------------------------------------------------------------- linear algebra
def solve(A, b):
    """Gaussian elimination with partial pivoting: solve A x = b."""
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12:
            raise ValueError("Singular matrix - features are collinear or too few rows")
        for r in range(n):
            if r != c:
                factor = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= factor * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def add_intercept(X):
    return [[1.0] + list(row) for row in X]


def ols_fit(X, y):
    """Ordinary least squares via normal equations. X must include the intercept column."""
    p = len(X[0])
    XtX = [[sum(row[a] * row[b] for row in X) for b in range(p)] for a in range(p)]
    Xty = [sum(row[a] * yi for row, yi in zip(X, y)) for a in range(p)]
    return solve(XtX, Xty)


def linear_predict(X, beta):
    return [sum(x * b for x, b in zip(row, beta)) for row in X]


def standardize(train, test):
    """Z-score features using training-set statistics only (avoids leakage)."""
    cols = range(len(train[0]))
    means = [sum(r[j] for r in train) / len(train) for j in cols]
    sds = [math.sqrt(sum((r[j] - means[j]) ** 2 for r in train) / len(train)) or 1.0 for j in cols]

    def apply(rows):
        return [[(r[j] - means[j]) / sds[j] for j in cols] for r in rows]

    return apply(train), apply(test)


def logistic_fit(X, y, lr=0.1, epochs=5000):
    """Logistic regression by batch gradient descent. X has no intercept column."""
    w = [0.0] * len(X[0])
    b = 0.0
    n = len(X)
    for _ in range(epochs):
        gw = [0.0] * len(w)
        gb = 0.0
        for xi, yi in zip(X, y):
            z = b + sum(wj * xj for wj, xj in zip(w, xi))
            p = 1 / (1 + math.exp(-z)) if z >= 0 else math.exp(z) / (1 + math.exp(z))
            err = p - yi
            gb += err
            for j, xj in enumerate(xi):
                gw[j] += err * xj
        w = [wj - lr * g / n for wj, g in zip(w, gw)]
        b -= lr * gb / n
    return w, b


def logistic_predict(X, w, b):
    return [1 if b + sum(wj * xj for wj, xj in zip(w, row)) >= 0 else 0 for row in X]


# ---------------------------------------------------------------- metrics
def regression_metrics(y, yhat):
    n = len(y)
    mean = sum(y) / n
    ss_res = sum((a - b) ** 2 for a, b in zip(y, yhat))
    ss_tot = sum((a - mean) ** 2 for a in y)
    r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
    rmse = math.sqrt(ss_res / n)
    mae = sum(abs(a - b) for a, b in zip(y, yhat)) / n
    return r2, rmse, mae


def classification_metrics(y, yhat):
    tp = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(y, yhat) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, yhat) if a == 1 and b == 0)
    acc = (tp + tn) / len(y)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return acc, prec, rec


def time_split(n, test_share=0.2):
    """Chronological split: the most recent years form the test set."""
    n_test = max(3, round(n * test_share))
    return n - n_test


def fmt(x):
    return "n/a" if x != x else f"{x:.3f}"


# ---------------------------------------------------------------- experiments
def run_models(series):
    report = []

    # 1) Simple linear regression: GDP per capita ~ year
    years, X, y = make_table(series, ["gdp_pc"], "gdp_pc")
    t = [[yr - years[0]] for yr in years]
    k = time_split(len(years))
    beta = ols_fit(add_intercept(t[:k]), y[:k])
    pred = linear_predict(add_intercept(t[k:]), beta)
    r2, rmse, mae = regression_metrics(y[k:], pred)
    report.append(("Simple linear (GDP pc ~ year)", years[0], years[-1], len(years), r2, rmse, mae,
                   f"intercept={beta[0]:.1f}, slope per year={beta[1]:.1f}"))

    # 2) Polynomial regression (degree 2): GDP pc ~ year + year^2
    tp = [[(yr - years[0]) / 10, ((yr - years[0]) / 10) ** 2] for yr in years]
    beta = ols_fit(add_intercept(tp[:k]), y[:k])
    pred = linear_predict(add_intercept(tp[k:]), beta)
    r2, rmse, mae = regression_metrics(y[k:], pred)
    report.append(("Polynomial degree 2 (GDP pc ~ year)", years[0], years[-1], len(years), r2, rmse, mae,
                   f"coefficients={[round(b, 1) for b in beta]}"))

    # 3) Multiple linear regression: GDP pc ~ several indicators
    feats = ["internet", "trade", "life_exp", "inflation", "unemployment"]
    years, X, y = make_table(series, feats, "gdp_pc")
    k = time_split(len(years))
    Xtr, Xte = standardize(X[:k], X[k:])
    beta = ols_fit(add_intercept(Xtr), y[:k])
    pred = linear_predict(add_intercept(Xte), beta)
    r2, rmse, mae = regression_metrics(y[k:], pred)
    coefs = ", ".join(f"{f}={b:.1f}" for f, b in zip(feats, beta[1:]))
    report.append(("Multiple linear (GDP pc ~ 5 indicators)", years[0], years[-1], len(years), r2, rmse, mae,
                   f"standardized coefficients: {coefs}"))

    # 4) Logistic regression: is inflation above 10% in a given year?
    feats = ["trade", "unemployment", "internet"]
    years, X, y_raw = make_table(series, feats, "inflation")
    y = [1 if v > 10 else 0 for v in y_raw]
    k = time_split(len(years))
    Xtr, Xte = standardize(X[:k], X[k:])
    w, b = logistic_fit(Xtr, y[:k])
    pred = logistic_predict(Xte, w, b)
    acc, prec, rec = classification_metrics(y[k:], pred)
    coefs = ", ".join(f"{f}={wj:.2f}" for f, wj in zip(feats, w))
    report.append(("Logistic (inflation > 10%)", years[0], years[-1], len(years), acc, prec, rec,
                   f"accuracy/precision/recall on test set; weights: {coefs}"))

    return report


def write_report(report):
    OUT.mkdir(exist_ok=True)
    path = OUT / "results.md"
    lines = [
        "# UzPredict - model results",
        "",
        "Chronological 80/20 split: the last ~20% of years form the test set.",
        "",
        "| Model | Years | Rows | Metric 1 | Metric 2 | Metric 3 | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, y0, y1, n, m1, m2, m3, note in report:
        lines.append(f"| {name} | {y0}-{y1} | {n} | {fmt(m1)} | {fmt(m2)} | {fmt(m3)} | {note} |")
    lines += [
        "",
        "Metric 1-3 meaning: regression models = R2 / RMSE / MAE (test set);",
        "logistic model = accuracy / precision / recall (test set).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved results: {path}")


def main():
    series = {name: load_series(code) for name, code in INDICATORS.items()}
    save_merged_csv(series)
    report = run_models(series)
    for row in report:
        print(row[0], "| metrics:", [fmt(v) for v in row[4:7]], "|", row[7])
    write_report(report)


if __name__ == "__main__":
    main()
