---
layout: post
title: "How to Deploy a Large Language Model from Hugging Face to AWS: Part 2 — Lambda Function Setup"
date: 2025-10-01
category: ml
---

# Connecting a Lambda Function to Your SageMaker Endpoint

This is Part 2 of a three-part series. In [Part 1](/part1_sagemaker_deployment_guide/) we deployed a Hugging Face LLM to a SageMaker endpoint. Here, we'll wire a Lambda function to that endpoint so inference requests can be handled serverlessly — no persistent compute, no idle costs.

**Before starting:** Have the SageMaker endpoint ARN from Part 1 ready. You'll paste it into an IAM policy below.

---

## Step 1: Create an IAM Role for Lambda

Lambda needs a role that grants it permission to call your SageMaker endpoint. Start by creating a role scoped to the Lambda service.

Go to **IAM → Roles → Create role**. Set the trusted entity to **AWS Service** and select **Lambda** as the use case, then click **Next**.

Don't attach any permissions yet — you'll create a custom policy in the next step and attach it before finalising the role.

---

## Step 2: Create and Attach a SageMaker Invoke Policy

Rather than granting broad SageMaker access, the Lambda role should only be permitted to invoke your specific endpoint. This follows least-privilege best practice and limits the blast radius if the role is ever misused.

On the permissions screen, click **Create policy** (this opens a new tab). Switch to the **JSON** editor and paste the following, replacing the `Resource` ARN with the one you saved from Part 1:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": ["sagemaker:InvokeEndpoint"],
            "Resource": ["arn:aws:sagemaker:us-east-2:YOUR_ACCOUNT_ID:endpoint/lamini-t5-gpu-endpoint"]
        }
    ]
}
```

Name the policy `SageMakerInvokePolicy` and create it. Return to the role creation tab, refresh the policy list, attach `SageMakerInvokePolicy`, and complete role creation with the name `generate-text-lamini-role`.

Once the role is created, open it and copy the **Role ARN** — you'll need it in the next step.

---

## Step 3: Create the Lambda Function

Go to **Lambda → Create function** and configure it as follows:

| Setting | Value |
|---|---|
| Author from scratch | — |
| Function name | `generate-text-lamini` |
| Runtime | Python 3.10 |
| Execution role | Use an existing role → `generate-text-lamini-role` |

Click **Create function**.

---

## Step 4: Add the Function Code

In the **Code** tab, replace the default handler with the following:

```python
import boto3
import json

client = boto3.client('sagemaker-runtime')
ENDPOINT_NAME = 'lamini-t5-gpu-endpoint'


def lambda_handler(event, context):
    try:
        prompt = event.get('prompt', '')

        if not prompt:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Prompt not provided'})
            }

        response = client.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps({"inputs": prompt})
        )

        result = json.loads(response['Body'].read().decode())

        return {
            'statusCode': 200,
            'body': json.dumps(result)
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

The handler expects a JSON event with a `prompt` key, forwards it to the SageMaker endpoint as the `inputs` field, and returns the model's response. Missing prompts return a `400` rather than letting the request fail silently downstream.

Click **Deploy** to save the function.

---

## Step 5: Set the Timeout

Lambda's default timeout is 3 seconds — far too short for LLM inference, which can easily take 10–30 seconds depending on prompt length and instance warmup. Go to **Configuration → General configuration → Edit** and raise the timeout to at least **60 seconds**.

While you're there, confirm the memory allocation. The default 128 MB is fine here since the heavy computation happens on the SageMaker instance, not in Lambda.

---

## Summary

At this point you have:

- A least-privilege IAM role allowing Lambda to invoke only your specific SageMaker endpoint
- A deployed Lambda function that accepts a `prompt`, calls the endpoint, and returns the model's output
- A timeout configuration that won't cut off inference mid-generation

**Next:** [Part 3 — API Gateway Setup: Exposing Your Lambda as a Public HTTP Endpoint](/part3_api_gateway_setup_guide/)
