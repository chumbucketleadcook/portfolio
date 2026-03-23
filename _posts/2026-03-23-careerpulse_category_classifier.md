---
layout: post
title: "CareerPulse: Building a Job Category Classifier with PySpark, Sentence Transformers, and MLflow"
date: 2026-03-23
category: projects
---
# Building a Job Category Classifier with PySpark, Sentence Transformers, and MLflow

## Overview

This article summarises the design decisions and implementation work behind the CareerPulse category classifier — a machine learning pipeline built on Databricks that ingests job postings from The Muse API, processes them through a medallion architecture, and uses a KNN classifier to impute missing job category labels. The work covered here spans the Gold feature layer, model training with experiment tracking, and the supporting data engineering decisions that make the pipeline reliable and maintainable.

---

## The Medallion Architecture

CareerPulse organises data into three layers — Bronze, Silver, and Gold — each with a distinct responsibility.

**Bronze** is a faithful, append-only record of the raw API payload. Every page fetched from The Muse API is stored as a JSON string alongside ingestion metadata. Nothing is cleaned or transformed here. The Bronze layer exists so that if a bug is introduced downstream, the original data is always recoverable.

**Silver** parses the raw JSON into a structured schema, applies data quality checks, and normalises sentinel values. One important normalisation introduced during this work was coercing the API's `"Unknown"` category value to `NULL`. The Muse API uses `"Unknown"` as a placeholder for missing categories — semantically equivalent to null — and allowing it to propagate downstream would have polluted both the Gold feature tables and the model training data. The fix was applied at the Silver transform using a case-insensitive guard:

```python
F.when(
    F.lower(F.get(col("job.categories"), 0).getField("name")).isin(["unknown", "none", ""]), None
).otherwise(
    F.get(col("job.categories"), 0).getField("name")
).alias("category")
```

Applying this at Silver rather than Gold ensures that all downstream consumers — demand aggregations, the labeled training set, and batch inference — naturally exclude these rows without any additional filtering logic.

**Gold** contains two types of tables: analytical aggregations used for demand trend analysis, and ML-ready feature tables used for model training and inference.

---

## Data Engineering Decisions

### Where Cleaning Lives

A recurring theme throughout this work was deciding which layer owns which transformation. The general principle applied was:

- **Data validity** (nulls, sentinel values, malformed records) belongs in Silver
- **ML suitability** (filtering thin categories, constructing features) belongs in Gold
- **Model logic** (train/test splits, hyperparameter sweeps) belongs in the training notebook

This separation keeps each layer's responsibility clear and makes the pipeline easier to debug — if a model produces unexpected predictions, you know exactly which layer to inspect.

### The `category_labeled_postings` Table

Rather than training directly from Silver, a dedicated Gold table called `category_labeled_postings` was created to serve as the clean, ML-ready input to the training notebook. This table contains only rows where `category` is not null, with HTML-stripped and normalised `description_clean` text, and deduplicates on `posting_id` to prevent class inflation during cross-validation.

The decision to name it `category_labeled_postings` rather than `knn_training_set` was deliberate — the table represents the full labeled population available for modeling, and the train/test split happens at runtime in the training notebook. Naming it after the model would have been a misnomer and would have made the table name stale if the model approach changed.

### MERGE Behaviour and Idempotency

Several bugs encountered during this work stemmed from misunderstanding Delta MERGE semantics. Key lessons:

- A MERGE keyed on `posting_id` with only `WHEN NOT MATCHED THEN INSERT` is idempotent — re-running the pipeline will never duplicate rows, but will also never update existing ones. This is the correct behaviour for Silver and Gold in steady state.
- Adding `WHEN MATCHED THEN UPDATE SET *` requires the source to have at most one row per key. The `DELTA_MULTIPLE_SOURCE_ROW_MATCHING_TARGET_ROW` error occurs when multiple source rows match the same target row — resolved by deduplicating the source with `dropDuplicates(["posting_id"])` before the MERGE.
- Historical rows written before a data quality fix are not retroactively corrected by re-running the pipeline with an insert-only MERGE. One-time `UPDATE` or `DELETE` statements are the right tool for backfilling corrections to existing rows.

### PySpark UDFs

HTML stripping and text normalisation were applied using a PySpark UDF wrapping a utility function from the `utils.clean_description` module:

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from utils.clean_description import clean_description

clean_description_udf = udf(clean_description, StringType())
```

PySpark does not have a native `apply()` equivalent. UDFs are the correct approach for arbitrary Python logic that cannot be expressed using native Spark functions, with the caveat that they serialise data out of the JVM to Python and back, which carries overhead. For this use case — BeautifulSoup HTML stripping, regex cleaning, whitespace normalisation — a regular UDF is appropriate. For high-throughput scenarios, `pandas_udf` with Arrow serialisation would be more efficient.

---

## The Category Classifier

### Problem Framing

A significant proportion of job postings in The Muse API arrive without a category label. The classifier's job is to impute these missing labels using the text of the job title and description. This is a multiclass text classification problem with roughly 20 categories ranging from `"Data & Analytics"` to `"Marketing"` to `"Engineering"`.

### Embedding Approaches Compared

Three embedding approaches were evaluated via stratified k-fold cross-validation:

| Approach | Description |
|---|---|
| TF-IDF + KNN | Sparse bag-of-words vectors, cosine distance |
| TF-IDF + SVD + KNN | Dimensionality reduction to 200 components via Latent Semantic Analysis |
| Sentence Transformer + KNN | Dense contextual embeddings from `all-MiniLM-L6-v2` |

Two additional modifications were evaluated on top of the sentence transformer approach:

- **Distance weighting** (`weights="distance"`) — closer neighbors contribute more to the prediction than distant ones
- **Title boosting** — prepending the job title `n` times before encoding to upweight title tokens in the embedding

The winning configuration was **Sentence Transformer + distance weighting + title prepended twice + k=7**, evaluated using macro F1 as the primary metric. Macro F1 was chosen over accuracy because it treats each class equally regardless of size — important given the class imbalance across job categories.

### Title Boosting

Title boosting was implemented as a vectorised pandas operation:

```python
def boost_title(df: pd.DataFrame, n: int) -> pd.Series:
    return (df["title"] + " ") * n + df["description_clean"]
```

The intuition is that job titles carry dense, high-signal text — `"Senior Data Engineer"` is more informative per token than most sentences in a job description. Prepending the title multiple times increases its weight in the sentence transformer's attention without requiring any architectural changes to the model.

### Cross-Validation Strategy

A `StratifiedKFold` splitter with 5 folds was used throughout. Stratification ensures that each fold contains a proportional representation of every class — important when some categories have significantly fewer examples than others. Both mean and standard deviation of F1 across folds were tracked: a high standard deviation signals that the model is sensitive to which data it sees, often a symptom of thin classes rather than a fundamental model problem.

---

## MLflow Experiment Tracking

### Experiment Structure

Each combination of embedding method and k value was logged as a separate MLflow run under a shared experiment. All runs within a single sweep share a `RUN_TIMESTAMP` defined once at the top of the notebook:

```python
from datetime import datetime
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
```

Defining the timestamp once rather than calling `datetime.now()` inside the loop ensures all runs from the same sweep are identifiable as a group in the MLflow UI.

### What Gets Logged

Each run logs:

- **Params** — embedding method, k, distance metric, weighting strategy, title prepending value, CV folds, test size
- **Metrics** — CV accuracy mean/std, CV macro F1 mean/std, test accuracy, test macro F1
- **Artifacts** — a per-class classification report showing precision, recall and F1 for every category

The per-class report is particularly valuable — aggregate metrics can mask regressions in specific categories, and having the full breakdown as an artifact on every run makes it easy to diagnose which classes are driving performance differences between configurations.

### Registering the Winning Model

The winning model cell was designed to be configuration-driven — changing `WINNING_METHOD` at the top of the cell dispatches to the correct pipeline construction and encoding logic, and everything downstream is method-agnostic:

```python
WINNING_METHOD        = "sentence_transformer"  # "tfidf" | "tfidf_svd" | "sentence_transformer"
WINNING_K             = 7
WINNING_WEIGHTS       = "distance"
WINNING_TITLE_REPEATS = 2
ST_MODEL_NAME         = "all-MiniLM-L6-v2"
```

Before registering, the estimator is fit twice — first on `X_train` to produce honest held-out test metrics, then on the full labeled dataset so the registered model has seen all available signal. The fully-fitted estimator is what gets saved to the MLflow Model Registry.

---

## Model Validation and Promotion

Registering a new model version should not mean immediately promoting it to Production. The recommended validation workflow before promotion:

1. Register the new version to **Staging**, not Production
2. Compare per-class F1 between the current Production version and the new Staging version — sort by worst regressions to catch cases where a higher aggregate F1 masks a specific category getting worse
3. Inspect predictions on recent Silver postings for sanity — correct class distribution, no unexpected nulls, no sentinel values
4. Only promote to Production after the validation gate passes

This validation logic is a natural fit for `06_monitoring`, where it can run automatically as part of the scheduled pipeline rather than requiring manual intervention on every retrain cycle.

---

## What's Next

With `04_train_models` complete and a registered model in the MLflow Model Registry, the next steps in the CareerPulse pipeline are:

- **`05_batch_inference`** — load the registered Production model, score null-category rows from Silver, write predictions to `careerpulse_inference.category_predictions`
- **`06_monitoring`** — automated model validation, prediction distribution tracking, and drift detection
- **`07_streamlit_app`** — a front-end interface surfacing Gold layer demand analytics and inference results
