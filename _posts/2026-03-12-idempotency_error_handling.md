---
layout: post
title: "Designing for Failure: Idempotency and Error Handling in PySpark Pipelines"
date: 2026-04-15
category: data
permalink: /idempotency-error-handling/
---
# Designing for Failure: Idempotency and Error Handling in PySpark Pipelines

## Overview

Most data engineering tutorials show you how to build a pipeline that works. Almost none of them show you what happens when it doesn't.

In production, pipelines fail. An upstream API returns a malformed payload. A Databricks job times out halfway through a write. A schema change in the source system sends a null where your code expected a string. How your pipeline behaves in each of these scenarios determines whether you have a data platform or a data liability.

This post covers the failure patterns I've encountered building production PySpark pipelines — at P&G and in personal projects like CareerPulse — and the design principles that make pipelines robust enough to fail gracefully, recover cleanly, and never corrupt the data they're responsible for.

---

## The Core Problem: Non-Idempotent Writes

An operation is **idempotent** if running it multiple times produces the same result as running it once. For data pipelines, this matters because pipelines get re-run — intentionally (backfills, reruns after fixing a bug) and unintentionally (job failures mid-execution, duplicate trigger events).

A naive PySpark pipeline appending to a Delta table is not idempotent:

```python
# ❌ Not idempotent — reruns will duplicate data
(
    transformed_df
    .write
    .format("delta")
    .mode("append")
    .saveAsTable("workspace.careerpulse_silver.job_postings")
)
```

If this job fails after a partial write and is restarted from the beginning, you get duplicate rows. If it succeeds but is accidentally triggered twice, same result. Downstream models and aggregations silently produce wrong answers.

The fix is to make writes explicitly idempotent by keying on a natural unique identifier:

```python
# ✅ Idempotent — safe to rerun
from delta.tables import DeltaTable

def upsert_to_silver(spark, new_data_df, table_name: str, merge_key: str = "posting_id"):
    """
    Merge new records into a Delta table using posting_id as the deduplication key.
    Inserts new records, updates existing ones. Safe to rerun.
    """
    target = DeltaTable.forName(spark, table_name)

    (
        target.alias("target")
        .merge(
            new_data_df.alias("source"),
            f"target.{merge_key} = source.{merge_key}"
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
```

Delta Lake's `MERGE` operation is atomic — it either completes fully or rolls back entirely, which means partial writes no longer leave your table in a broken intermediate state.

---

## Handling Malformed API Payloads

In CareerPulse, data arrives from The Muse API as raw JSON. Not every payload is clean. Fields go missing, types change, nested structures occasionally arrive flat. The Bronze layer's job is to land raw data faithfully, but the Silver transformation step needs to handle variance without crashing.

The pattern I use is **schema validation with a quarantine path** — records that fail validation are routed to a separate bad-records table rather than halting the entire job:

```python
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
import json

EXPECTED_FIELDS = {"id", "title", "company", "category", "description", "levels", "locations"}

def validate_and_split(df):
    """
    Split a Bronze DataFrame into valid records and quarantined bad records.
    Returns (valid_df, quarantine_df).
    """
    def is_valid(json_str):
        try:
            record = json.loads(json_str)
            missing = EXPECTED_FIELDS - set(record.keys())
            return len(missing) == 0
        except Exception:
            return False

    is_valid_udf = F.udf(is_valid, "boolean")

    valid_df = df.filter(is_valid_udf(F.col("raw_payload")))
    quarantine_df = (
        df.filter(~is_valid_udf(F.col("raw_payload")))
          .withColumn("quarantine_reason", F.lit("missing_required_fields"))
          .withColumn("quarantined_at", F.current_timestamp())
    )

    return valid_df, quarantine_df


def write_with_quarantine(spark, df, silver_table: str, quarantine_table: str):
    valid_df, bad_df = validate_and_split(df)

    if bad_df.count() > 0:
        (
            bad_df.write.format("delta")
            .mode("append")
            .saveAsTable(quarantine_table)
        )
        print(f"Quarantined {bad_df.count()} bad records to {quarantine_table}")

    upsert_to_silver(spark, valid_df, silver_table)
```

The quarantine table is not a graveyard — it's an audit log. When the upstream API changes its schema, the quarantine table tells you exactly which records were affected and when, giving you everything you need to write a backfill once the parsing logic is updated.

---

## Checkpoint-Based Recovery for Long-Running Jobs

For pipelines that process large historical windows, failing halfway through and restarting from scratch is expensive. Checkpointing lets you resume from the last successful partition rather than from the beginning.

```python
from datetime import date, timedelta
import json
import os

CHECKPOINT_PATH = "/dbfs/careerpulse/checkpoints/bronze_ingestion.json"

def load_checkpoint() -> str:
    """Load the last successfully processed date from checkpoint file."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            data = json.load(f)
            return data.get("last_successful_date")
    return None


def save_checkpoint(processed_date: str):
    """Persist the last successfully processed date."""
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"last_successful_date": processed_date}, f)


def run_incremental_ingestion(spark, start_date: str = None, end_date: str = None):
    """
    Run Bronze ingestion only for dates not yet processed.
    Resumes from checkpoint on restart.
    """
    checkpoint = load_checkpoint()
    effective_start = checkpoint or start_date or "2024-01-01"
    effective_end = end_date or date.today().isoformat()

    print(f"Processing {effective_start} → {effective_end}")

    current = date.fromisoformat(effective_start)
    end = date.fromisoformat(effective_end)

    while current <= end:
        date_str = current.isoformat()
        try:
            ingest_partition(spark, date_str)   # your per-date ingestion logic
            save_checkpoint(date_str)
            print(f"✓ {date_str}")
        except Exception as e:
            print(f"✗ Failed on {date_str}: {e}")
            raise   # re-raise so the Databricks job marks as failed

        current += timedelta(days=1)
```

The key design decision here is raising the exception after logging rather than swallowing it. Silently continuing past a failed partition means you'll have a gap in your data with no visible signal — the worst possible failure mode in a time series pipeline.

---

## Schema Evolution: Don't Let Source Changes Break Production

Source schemas change. A field gets renamed, a new nested object is added, a type changes from `string` to `integer`. Without a schema management strategy, any of these will crash your Silver transformation with a cryptic error.

Delta Lake's schema evolution features handle the common cases gracefully, but you need to opt into them explicitly:

```python
# Allow new columns to be added without breaking the write
(
    transformed_df
    .write
    .format("delta")
    .mode("append")
    .option("mergeSchema", "true")   # automatically add new columns
    .saveAsTable("workspace.careerpulse_silver.job_postings")
)
```

`mergeSchema` handles additive changes — new columns — safely. What it won't protect you from is destructive changes: a column being dropped, or a type changing incompatibly. For those, a pre-write schema check that compares incoming columns against the registered table schema gives you an early, meaningful error rather than a confusing downstream failure:

```python
def assert_required_columns_present(df, required_columns: list):
    """Raise early if any expected column is missing from the DataFrame."""
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(
            f"Schema validation failed. Missing columns: {missing}. "
            f"Check upstream API for schema changes."
        )
```

---

## Structured Logging for Observability

When a production pipeline fails at 3am, the quality of your logs determines how quickly you can diagnose it. Unstructured `print()` calls scattered through notebooks are not sufficient. Structured logging — consistent keys, machine-parseable format — lets you query your logs and build alerts around them.

```python
import json
import logging
from datetime import datetime

logger = logging.getLogger("careerpulse")

def log_pipeline_event(stage: str, status: str, records_processed: int = None, error: str = None):
    """Emit a structured log event for a pipeline stage."""
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "pipeline": "careerpulse",
        "stage": stage,
        "status": status,
    }
    if records_processed is not None:
        event["records_processed"] = records_processed
    if error is not None:
        event["error"] = error

    logger.info(json.dumps(event))


# Usage
log_pipeline_event("silver_transform", "started")
try:
    upsert_to_silver(spark, transformed_df, "workspace.careerpulse_silver.job_postings")
    log_pipeline_event("silver_transform", "completed", records_processed=transformed_df.count())
except Exception as e:
    log_pipeline_event("silver_transform", "failed", error=str(e))
    raise
```

In Databricks, these structured JSON logs flow into CloudWatch or the Databricks logging backend where they can be queried, dashboarded, and used to trigger alerts — turning your pipeline's operational state from invisible to observable.

---

## The Mental Model: Pipelines as Contracts

The underlying principle across all of these patterns is that a production pipeline is a **contract** with its downstream consumers. It promises to deliver complete, deduplicated, correctly typed data on a reliable schedule. Every defensive pattern described here — idempotent writes, schema validation, quarantine paths, checkpointing, structured logging — is a mechanism for honoring that contract even when the world outside your pipeline behaves badly.

Tutorials rarely cover this because it's unglamorous. But in a hiring context, being able to speak fluently about how you handle failure is one of the clearest signals that you've actually run things in production rather than just built them to work the first time.

---

## Key Takeaways

- Use Delta Lake `MERGE` rather than `append` for any table that must survive reruns
- Route bad records to a quarantine table rather than dropping them or halting the job
- Checkpoint long-running jobs at the partition level so failures restart cheaply
- Opt into `mergeSchema` for additive schema evolution, and add explicit checks for destructive changes
- Emit structured JSON logs from every pipeline stage so failures are diagnosable, not just visible

---

*CareerPulse · Data Pipeline Documentation · 2026*
