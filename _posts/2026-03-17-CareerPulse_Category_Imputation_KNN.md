---
layout: post
title: "CareerPulse: Job Category Imputation Using KNN"
date: 2026-03-17
category: ml
---
# Predicting Job Categories with KNN: A CareerPulse Case Study

## Overview

The CareerPulse pipeline ingests job postings from [The Muse API](https://www.themuse.com/developers/api/v2) and transforms them through a medallion architecture (Bronze → Silver → Gold) before feeding a forecasting model. One of the first classification challenges encountered was **predicting the job `category` field** — the primary label used to aggregate daily posting counts and build demand-forecasting features.

Because The Muse API returns category labels for most postings, the labeled data makes this a supervised learning problem well-suited for a K-Nearest Neighbors (KNN) classifier.

---

## The Problem

After parsing the Bronze payload and flattening it into the Silver table (`careerpulse_silver.job_postings`), each row carries:

- `title` — the raw job title string
- `description` — full HTML job description
- `category` — a label like `"Data Science"`, `"Software Engineering"`, `"Marketing"`, etc.

A small but non-trivial share of postings arrive with a `NULL` category. Rather than drop those rows — which would create gaps in the daily time series and degrade the downstream feature quality — the goal was to impute missing categories using the text features available on every posting.

---

## Data Preparation

The feature matrix was built from the Silver table using PySpark and then collected to a Pandas DataFrame for scikit-learn training:

```python
from pyspark.sql import functions as F

labeled = (
    spark.table("workspace.careerpulse_silver.job_postings")
    .filter(F.col("category").isNotNull())
    .select("posting_id", "title", "description", "category")
    .dropDuplicates(["posting_id"])
)
```

Text from `title` and `description` was combined into a single document per row, then vectorized using **TF-IDF** with a max vocabulary of 10,000 tokens and English stop-word removal:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

df["text"] = df["title"].fillna("") + " " + df["description"].str.replace(r"<[^>]+>", "", regex=True).fillna("")

vectorizer = TfidfVectorizer(max_features=10_000, stop_words="english", sublinear_tf=True)
X = vectorizer.fit_transform(df["text"])
y = df["category"]
```

HTML tags were stripped from `description` before vectorization. `sublinear_tf=True` applies log-normalization to term frequencies, which helps compress the influence of very common words across long job descriptions.

---

## Model Training

A standard train/test split (80/20, stratified by category) was used to preserve class proportions across the relatively imbalanced label distribution:

```python
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

knn = KNeighborsClassifier(n_neighbors=7, metric="cosine", n_jobs=-1)
knn.fit(X_train, y_train)
```

**Cosine distance** was chosen over the default Euclidean metric because TF-IDF vectors live in a high-dimensional sparse space where cosine similarity better captures topical relatedness between documents regardless of document length.

`k=7` was selected via 5-fold cross-validation across `k ∈ {3, 5, 7, 10, 15}`, with cosine distance consistently outperforming Euclidean at every value of `k`.

---

## Results

On the held-out test set the model achieved:

| Metric | Score |
|---|---|
| Accuracy | 0.83 |
| Macro F1 | 0.79 |
| Weighted F1 | 0.84 |

Performance was strongest on high-volume categories (`Software Engineering`, `Data Science`, `Marketing`) where the training signal was richest. Rarer categories like `Legal` and `Real Estate` showed lower recall, primarily due to class imbalance and overlap with adjacent categories in the TF-IDF embedding space.

A confusion matrix revealed the most common error was confusing `"Data & Analytics"` with `"Data Science"` — both categories share a large vocabulary of domain-specific terms (`Python`, `SQL`, `modeling`, `pipeline`), making them genuinely difficult to separate without richer structural features.

---

## Integration into the Pipeline

The trained vectorizer and classifier were serialized with `joblib` and registered in the Databricks workspace:

```python
import joblib, mlflow

with mlflow.start_run(run_name="knn_category_classifier"):
    mlflow.log_params({"k": 7, "metric": "cosine", "max_features": 10_000})
    mlflow.log_metrics({"accuracy": 0.83, "macro_f1": 0.79})
    mlflow.sklearn.log_model(knn, "knn_model")
    mlflow.log_artifact("tfidf_vectorizer.pkl")
```

Batch inference (`05_batch_inference`) loads the registered model, applies it to the `NULL`-category rows from Silver, and writes the imputed labels back — ensuring that `03_features_gold` can always produce a complete, gap-free daily time series for every category.

---

## Limitations & Next Steps

- **Class imbalance**: Minority categories would benefit from oversampling or a class-weighted loss function. A follow-up experiment replacing KNN with a logistic regression classifier (which natively supports `class_weight="balanced"`) showed a modest +2pp macro F1 improvement.
- **Feature richness**: The current model uses text alone. Adding `level` (seniority) and `location` as categorical features, or incorporating sentence embeddings via a lightweight transformer, are the most promising paths to higher accuracy.
- **Model drift**: As The Muse API's category taxonomy evolves, the classifier should be retrained periodically. The `06_monitoring` notebook is the intended home for category-distribution drift checks.

---

*CareerPulse · Data Pipeline Documentation · March 2026*