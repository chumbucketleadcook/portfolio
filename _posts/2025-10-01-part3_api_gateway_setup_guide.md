---
layout: post
title: "How to Deploy a Large Language Model from Hugging Face to AWS: Part 3 — API Gateway Setup"
date: 2025-10-01
category: ml
---

# Exposing Your Lambda Function as a Public HTTP Endpoint

This is the final part of the series. In [Part 1](/part1_sagemaker_deployment_guide/) we deployed a Hugging Face LLM to SageMaker, and in [Part 2](/part2_lambda_setup_guide/) we wired a Lambda function to it. Here we'll put an HTTP API in front of that Lambda using API Gateway, giving the outside world a clean endpoint to send prompts to.

---

## Step 1: Create an HTTP API

Go to **API Gateway → APIs → Create API**. When prompted to choose an API type, select **HTTP API** and click **Build**.

HTTP API is the right choice here — it's faster and cheaper than REST API, and the feature set is sufficient for a single-route inference endpoint.

---

## Step 2: Add the Lambda Integration

On the integrations screen, configure the following:

| Setting | Value |
|---|---|
| Integration type | Lambda function |
| Lambda function | `generate-text-lamini` |
| Add permissions to Lambda function | ✓ Checked |

Checking **Add permissions to Lambda function** lets API Gateway automatically configure the resource-based policy needed to invoke your Lambda — one less manual step. Click **Next**.

---

## Step 3: Configure the Route

Define a single route for inference requests:

| Setting | Value |
|---|---|
| Method | POST |
| Resource path | `/generate` |

Click **Next**.

---

## Step 4: Configure the Stage

Leave the stage name as **$default** and ensure **Auto-deploy** is enabled. Auto-deploy means any future changes to the API are published immediately without a manual deployment step.

Click **Next → Create**.

---

## Step 5: Retrieve the Invoke URL

After creation you'll land on the API dashboard. The **Invoke URL** is displayed at the top and follows this format:

```
https://your-api-id.execute-api.us-east-2.amazonaws.com
```

Append `/generate` to get your full endpoint:

```
https://your-api-id.execute-api.us-east-2.amazonaws.com/generate
```

Save this URL — it's the address you'll hit to run inference from any external client.

---

## Step 6: Test the Endpoint

Before wiring this into anything else, verify the full pipeline end-to-end. API Gateway has a built-in test console: go to **APIs → your API → Routes → POST /generate → Test** and send the following body:

```json
{
  "prompt": "Write a poem about the ocean."
}
```

A successful response confirms that API Gateway is correctly routing to Lambda, and Lambda is correctly invoking the SageMaker endpoint.

If the test times out, double-check that your Lambda timeout is set to at least 60 seconds (covered in Part 2). If you receive a `403`, verify that the Lambda resource-based policy includes an `Allow` for `apigateway.amazonaws.com` — this should have been set automatically in Step 2, but it's worth confirming under **Lambda → Configuration → Permissions**.

---

## Step 7: Call the API from Python

With the endpoint live, you can query your model from any HTTP client. Here's a minimal Python example:

```python
import requests

url = "https://your-api-id.execute-api.us-east-2.amazonaws.com/generate"

response = requests.post(url, json={"prompt": "Explain what a transformer model is."})
print(response.json())
```

Replace the URL with your actual Invoke URL from Step 5. The `json=` parameter in `requests.post` handles serialization and sets the correct `Content-Type` header automatically.

---

## Monitoring

Logs for both API Gateway and Lambda are written to **CloudWatch Logs** automatically. If a request succeeds at the API layer but returns an unexpected result, the Lambda log group (`/aws/lambda/generate-text-lamini`) is the first place to look.

---

## Summary

The three-part pipeline is now complete:

```
Client → API Gateway (POST /generate) → Lambda → SageMaker Endpoint → Model
```

You have a public HTTPS endpoint that accepts a JSON prompt, routes it through a serverless Lambda function, invokes a GPU-backed SageMaker endpoint, and returns the model's response — with no persistent compute running when the API is idle.

**From Part 1:** IAM execution role + SageMaker GPU endpoint serving LaMini-T5-738M  
**From Part 2:** Lambda function with least-privilege IAM role and a 60-second timeout  
**From Part 3:** HTTP API Gateway with auto-deploy routing `POST /generate` to Lambda
