---
layout: post
title: "From Prompt to Response in Under a Second: Optimizing a Serverless LLM Inference Pipeline"
date: 2026-05-01
category: ml
---
# From Prompt to Response in Under a Second: Optimizing a Serverless LLM Inference Pipeline

## Overview

In the [three-part series on deploying an LLM to AWS](/part1_sagemaker_deployment_guide/), the pipeline went from zero to a working public endpoint: Hugging Face model on SageMaker, Lambda function as the backend, API Gateway routing HTTP traffic. The architecture worked.

But "working" and "production-ready" are different things. On first deployment, end-to-end latency on the first request after a period of inactivity regularly exceeded 8–12 seconds. Subsequent warm requests settled around 1.2–1.8 seconds. For most applications, both of those numbers need to come down significantly.

This post covers the specific optimizations applied to each layer of the stack — Lambda, SageMaker, and API Gateway — and how to measure the impact of each change so you're tuning against data rather than intuition.

---

## Establishing a Baseline

Before optimizing anything, measure everything. It's easy to spend time on the wrong bottleneck.

The end-to-end request lifecycle breaks into three distinct segments:

```
Client → API Gateway → Lambda → SageMaker Endpoint → Lambda → API Gateway → Client
         [  API GW  ]  [ Lambda cold start + execution ]  [ SageMaker inference ]
```

Use CloudWatch to pull timing data for each segment independently. Lambda reports `Init Duration` (cold start) and `Duration` (execution) separately in its logs. SageMaker exposes `ModelLatency` and `OverheadLatency` as CloudWatch metrics on the endpoint. The gap between your API Gateway access log timestamps and Lambda's reported duration is your Lambda invocation overhead.

```python
import boto3
from datetime import datetime, timedelta

cw = boto3.client('cloudwatch', region_name='us-east-2')

def get_sagemaker_latency(endpoint_name: str, hours: int = 1) -> dict:
    """Pull mean ModelLatency and OverheadLatency from CloudWatch."""
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)

    metrics = {}
    for metric_name in ["ModelLatency", "OverheadLatency"]:
        response = cw.get_metric_statistics(
            Namespace="AWS/SageMaker",
            MetricName=metric_name,
            Dimensions=[{"Name": "EndpointName", "Value": endpoint_name}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=["Average", "p99"]
        )
        if response["Datapoints"]:
            dp = response["Datapoints"][0]
            metrics[metric_name] = {
                "avg_ms": round(dp["Average"] / 1000, 2),  # microseconds → ms
                "p99_ms": round(dp["Maximum"] / 1000, 2)
            }

    return metrics
```

Run this before and after each optimization. The goal is to know exactly which segment you improved and by how much.

---

## Problem 1: Lambda Cold Starts

Lambda functions that haven't been invoked recently are "cold" — the execution environment needs to be initialized before the function runs. For Python functions with heavyweight imports (`boto3`, `json`, etc.), this can add 800ms–2s to the first request.

### Fix 1a: Move all imports to module level

The most common cold-start antipattern is importing inside the handler function. Imports at module level are executed once during initialization, not on every invocation:

```python
# ❌ Imports inside handler — re-executed on every call
def lambda_handler(event, context):
    import boto3
    import json
    client = boto3.client('sagemaker-runtime')
    # ...

# ✅ Imports and client initialization at module level — executed once
import boto3
import json

client = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = 'lamini-t5-gpu-endpoint'

def lambda_handler(event, context):
    # client is already initialized
    # ...
```

This alone reduced cold start duration in my setup by approximately 400ms.

### Fix 1b: Provisioned Concurrency

For latency-sensitive applications, Lambda's Provisioned Concurrency keeps a specified number of execution environments pre-initialized and warm at all times. Cold starts become effectively zero for those environments.

```python
lambda_client = boto3.client('lambda', region_name='us-east-2')

response = lambda_client.put_provisioned_concurrency_config(
    FunctionName='generate-text-lamini',
    Qualifier='$LATEST',
    ProvisionedConcurrentExecutions=2   # keep 2 environments warm
)

print(response['Status'])  # 'IN_PROGRESS' → 'READY'
```

The tradeoff is cost — you're billed for provisioned concurrency even when the function isn't being invoked. For a low-traffic personal project, it's hard to justify. For a production API with SLA requirements, it's the right call.

### Fix 1c: Increase Lambda Memory

Lambda allocates CPU proportionally to memory. A 256MB Lambda function gets half the CPU of a 512MB function. For functions doing non-trivial JSON processing, doubling memory often meaningfully reduces execution duration — and because you're billed on duration × memory, the cost impact is often neutral or positive.

Test at 256MB, 512MB, and 1024MB. The sweet spot for this inference proxy is typically 512MB — beyond that, the bottleneck shifts entirely to SageMaker and additional Lambda memory produces diminishing returns.

---

## Problem 2: SageMaker Endpoint Latency

The model is on a `ml.g4dn.xlarge` — a GPU instance with a T4. At 738M parameters, the LaMini-T5 model should be generating responses in 200–600ms depending on output length. If you're seeing numbers significantly above that, the likely culprits are payload serialization overhead or inference configuration.

### Fix 2a: Constrain Output Length

The single most impactful inference-side optimization is setting an explicit `max_new_tokens` limit. Without it, the model will generate until it hits its default maximum, which for text2text models can be far longer than your application needs:

```python
response = client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType='application/json',
    Body=json.dumps({
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,      # set based on your use case
            "do_sample": False,          # greedy decoding is faster than sampling
            "temperature": 1.0
        }
    })
)
```

Setting `do_sample=False` switches to greedy decoding, which is deterministic and slightly faster than sampling-based generation. For most API use cases where you want consistent, fast responses rather than creative variation, greedy decoding is the right default.

### Fix 2b: SageMaker Endpoint Autoscaling

A single-instance endpoint is a single point of failure and a concurrency bottleneck. Under any sustained load, requests will queue. Autoscaling adds instances in response to load and removes them when traffic drops:

```python
aas_client = boto3.client('application-autoscaling', region_name='us-east-2')

# Register the endpoint as a scalable target
aas_client.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/lamini-t5-gpu-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=3
)

# Define the scaling policy
aas_client.put_scaling_policy(
    PolicyName='lamini-endpoint-scaling',
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/lamini-t5-gpu-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 5.0,   # target: 5 concurrent requests per instance
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
        },
        'ScaleInCooldown': 300,
        'ScaleOutCooldown': 60    # scale out faster than scaling in
    }
)
```

Note the asymmetry in cooldown periods — scaling out (adding capacity) uses a 60-second cooldown while scaling in (removing instances) uses 300 seconds. This is intentional: you want to add capacity quickly under load and remove it conservatively to avoid oscillation.

---

## Problem 3: API Gateway Configuration

API Gateway adds minimal latency when configured correctly, but a few settings can quietly add overhead.

### Fix 3a: Enable Payload Compression

For larger prompts and longer model responses, enabling response compression at the API Gateway level reduces transfer size and improves perceived latency:

```python
apigw_client = boto3.client('apigatewayv2', region_name='us-east-2')

apigw_client.update_api(
    ApiId='YOUR_API_ID',
    MinimumCompressionSize=1024   # compress responses larger than 1KB
)
```

### Fix 3b: Add Throttling

Without throttling, a burst of requests will queue at Lambda and SageMaker in ways that degrade latency for all concurrent users. Setting rate and burst limits protects the endpoint's latency profile under load:

```python
apigw_client.update_stage(
    ApiId='YOUR_API_ID',
    StageName='$default',
    DefaultRouteSettings={
        'ThrottlingBurstLimit': 20,    # max concurrent requests
        'ThrottlingRateLimit': 10.0    # requests per second steady state
    }
)
```

Clients that exceed the limits receive a `429 Too Many Requests` response with a `Retry-After` header — a clean, expected failure mode rather than a timeout.

---

## Benchmark Results

After applying the optimizations above, here's how the numbers moved:

| Metric | Before | After | Improvement |
|---|---|---|---|
| Cold start latency | ~10s | ~1.2s | 88% |
| Warm request P50 | 1.6s | 0.7s | 56% |
| Warm request P99 | 3.1s | 1.4s | 55% |
| SageMaker ModelLatency P50 | 890ms | 420ms | 53% |

The cold start improvement came almost entirely from Provisioned Concurrency. The warm request improvement came primarily from the `max_new_tokens` constraint and moving to greedy decoding. The Lambda memory increase from 256MB to 512MB contributed a modest ~80ms reduction in execution duration.

---

## What to Tackle Next

- **Response streaming**: The Hugging Face TGI container supports token streaming, which dramatically improves perceived latency for long responses by starting to return tokens before generation completes. This requires changes at the SageMaker, Lambda, and API Gateway layers but is the single biggest UX improvement available.
- **Caching**: For a demo or query-heavy application, caching frequent prompts with ElastiCache or DynamoDB can serve repeated requests in <5ms.
- **Model quantization**: Quantizing the LaMini-T5 model to INT8 with `bitsandbytes` would reduce memory footprint and potentially allow a smaller (cheaper) instance type without meaningful quality degradation at this parameter scale.

---

*AWS · SageMaker · Lambda · API Gateway · MLOps · 2026*
