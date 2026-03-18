---
layout: post
title: "Monitoring ML Models in Production: Detecting Data Drift Before It Breaks Your Pipeline"
date: 2026-04-01
category: ml
permalink: /monitoring-model-drift/
---
# Monitoring ML Models in Production: Detecting Data Drift Before It Breaks Your Pipeline

## Overview

Deploying a model is the easy part. The harder problem is knowing when it stops working.

Most ML pipelines fail quietly. Accuracy doesn't crash overnight — it erodes. The job posting distribution shifts. A new category gains traction. The language in job titles evolves. Your model keeps predicting, the pipeline keeps running, and nobody notices until a stakeholder asks why the forecasts have been off for three weeks.

This post covers the drift monitoring layer I built for CareerPulse and the general principles behind it — what types of drift to watch for, which statistical tests are actually useful in practice, and how to build alerting that fires before downstream model quality degrades.

---

## The Three Types of Drift That Will Actually Hurt You

Before writing any code, it helps to be precise about what you're monitoring. There are three distinct failure modes:

**Feature drift (covariate shift)**: The distribution of input features changes. In CareerPulse, this would be the vocabulary in job titles and descriptions shifting over time — new frameworks entering the lexicon, old ones fading out.

**Label drift (prior probability shift)**: The distribution of the target variable changes. For CareerPulse this is the category label distribution — if `"AI & Machine Learning"` postings triple in six months, a model trained on older data will underpredict it.

**Concept drift**: The relationship between features and labels changes. The same title — say, `"Data Analyst"` — might have mapped cleanly to one category two years ago but now straddles two. This is the hardest to detect statistically because inputs and outputs can both look normal while the mapping between them quietly degrades.

For CareerPulse, **label drift** is the highest-risk failure mode because category distribution is directly what the downstream forecasting model is trying to predict. That's where monitoring effort should be concentrated first.

---

## Setting a Baseline

Drift is always measured relative to a reference distribution. Before writing any monitoring logic, establish a stable baseline from your training window:

```python
from pyspark.sql import functions as F
import pandas as pd

# Pull category distribution from the training period
baseline_df = (
    spark.table("workspace.careerpulse_silver.job_postings")
    .filter(F.col("ingestion_date").between("2024-01-01", "2024-12-31"))
    .filter(F.col("category").isNotNull())
    .groupBy("category")
    .agg(F.count("*").alias("count"))
    .withColumn("total", F.sum("count").over(Window.rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)))
    .withColumn("proportion", F.col("count") / F.col("total"))
    .drop("total")
)

baseline = baseline_df.toPandas().set_index("category")["proportion"].to_dict()
```

Persist this baseline somewhere stable — a Delta table, a JSON artifact in MLflow, or a versioned file in object storage. It needs to survive model retraining cycles.

---

## Population Stability Index (PSI)

PSI is borrowed from credit risk modeling and is one of the most practical tools for monitoring categorical distributions in production. It compares a current distribution against the baseline and produces a single scalar score:

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

The interpretation thresholds are well-established:

| PSI Value | Interpretation |
|---|---|
| < 0.10 | No significant shift |
| 0.10 – 0.20 | Moderate shift, investigate |
| > 0.20 | Major shift, retrain likely needed |

Here's a clean implementation:

```python
import numpy as np

def compute_psi(baseline: dict, current: dict, epsilon: float = 1e-6) -> float:
    """
    Compute Population Stability Index between two category distributions.
    
    Args:
        baseline: dict mapping category -> proportion (reference period)
        current:  dict mapping category -> proportion (current period)
        epsilon:  small smoothing value to avoid log(0)
    
    Returns:
        PSI score (float)
    """
    categories = set(baseline.keys()) | set(current.keys())
    psi = 0.0

    for cat in categories:
        expected = baseline.get(cat, epsilon)
        actual = current.get(cat, epsilon)
        psi += (actual - expected) * np.log(actual / expected)

    return psi


def get_current_distribution(spark, lookback_days: int = 30) -> dict:
    from pyspark.sql import functions as F
    from datetime import date, timedelta

    cutoff = (date.today() - timedelta(days=lookback_days)).isoformat()

    df = (
        spark.table("workspace.careerpulse_silver.job_postings")
        .filter(F.col("ingestion_date") >= cutoff)
        .filter(F.col("category").isNotNull())
        .groupBy("category")
        .count()
        .toPandas()
    )

    total = df["count"].sum()
    return dict(zip(df["category"], df["count"] / total))
```

---

## Kolmogorov-Smirnov Test for Continuous Features

PSI works well for categorical distributions. For continuous features — like TF-IDF document length or model confidence scores — the two-sample KS test is more appropriate:

```python
from scipy.stats import ks_2samp
import numpy as np

def check_confidence_drift(baseline_scores: np.ndarray, current_scores: np.ndarray, alpha: float = 0.05) -> dict:
    """
    Run KS test to detect drift in model prediction confidence scores.
    
    Returns dict with statistic, p_value, and a boolean drift flag.
    """
    stat, p_value = ks_2samp(baseline_scores, current_scores)

    return {
        "ks_statistic": round(stat, 4),
        "p_value": round(p_value, 4),
        "drift_detected": p_value < alpha
    }
```

Tracking prediction confidence over time is particularly useful as a leading indicator — if the model is becoming less confident on average, feature drift is likely the cause even if the labels haven't visibly shifted yet.

---

## Wiring It Into the Pipeline

The monitoring logic lives in a dedicated `06_monitoring` notebook that runs on a weekly schedule in Databricks. The structure is straightforward:

```python
# 06_monitoring.py (simplified)

import json
from datetime import date

# Load persisted baseline
with open("/dbfs/careerpulse/monitoring/category_baseline.json") as f:
    baseline = json.load(f)

# Compute current distribution
current = get_current_distribution(spark, lookback_days=30)

# Run PSI check
psi_score = compute_psi(baseline, current)

# Log results to a monitoring Delta table
monitoring_record = {
    "run_date": date.today().isoformat(),
    "psi_score": round(psi_score, 4),
    "alert": psi_score > 0.10
}

(
    spark.createDataFrame([monitoring_record])
    .write.format("delta")
    .mode("append")
    .saveAsTable("workspace.careerpulse_monitoring.drift_log")
)

# Raise alert if threshold exceeded
if psi_score > 0.20:
    raise ValueError(f"CRITICAL: Category distribution PSI = {psi_score:.3f}. Retraining required.")
elif psi_score > 0.10:
    print(f"WARNING: Moderate category drift detected (PSI = {psi_score:.3f}). Monitor closely.")
else:
    print(f"OK: PSI = {psi_score:.3f}. Distribution stable.")
```

Persisting results to a Delta table gives you a queryable audit trail. You can plot PSI over time in a notebook or connect it to a BI tool to visualize the trend before any single threshold is breached.

---

## Why Retraining Schedules Alone Aren't Enough

A common pattern is to retrain on a fixed cadence — monthly, quarterly — regardless of whether drift has actually occurred. This creates two problems simultaneously: you retrain unnecessarily when the distribution is stable (wasting compute and potentially introducing instability), and you retrain too late when drift accelerates unexpectedly.

Drift-triggered retraining solves both. The monitoring notebook acts as a circuit breaker — it either passes a green signal downstream or escalates, and retraining is initiated only when the data actually justifies it.

The other gap in schedule-based approaches is that they say nothing about *what* drifted. A PSI breakdown by category tells you not just that the distribution shifted, but which specific categories moved and by how much — which is actionable information when deciding whether to retrain the full classifier or just augment training data for the affected classes.

---

## Limitations & Next Steps

- **New categories**: The current PSI implementation handles unseen categories via epsilon smoothing, but a proper new-category detection step — flagging when a label outside the known taxonomy appears in volume — would be a cleaner solution.
- **Alerting integration**: The `raise ValueError` approach works for Databricks job failure alerts but a more robust setup would push to Slack or PagerDuty via webhook.
- **Model performance tracking**: PSI monitors inputs and label distributions, but tracking held-out accuracy on a labeled validation window (when labels are available with a delay) is the gold-standard complement to distribution monitoring.

---

*CareerPulse · Data Pipeline Documentation · 2026*
