---
layout: post
title: "How to Deploy a Large Language Model from Hugging Face to AWS: Part 1 — SageMaker"
date: 2025-10-01
category: ml
---

# Deploying a Hugging Face LLM to a SageMaker Endpoint

This is Part 1 of a three-part series on building a fully serverless LLM inference pipeline on AWS. By the end of this post, you'll have a live SageMaker endpoint serving a Hugging Face model — ready for the Lambda and API Gateway layers covered in Parts 2 and 3.

**What you'll need:** An AWS account with permission to create IAM roles, SageMaker resources, and billing enabled for GPU instances.

---

## Step 1: Create an IAM Execution Role

SageMaker needs an execution role to access AWS resources on your behalf — S3 for model artifacts, ECR for container images, and so on.

Navigate to **IAM → Roles → Create role**. Set the trusted entity to **SageMaker**, then attach two managed policies:

- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

Name the role `SageMakerExecutionRole` and create it. Once created, open the role and copy the **Role ARN** — you'll paste it into the deployment script in Step 5.

---

## Step 2: Launch a SageMaker Notebook Instance

The notebook instance is where you'll run the Python deployment code.

Go to **SageMaker → Notebook instances → Create notebook instance** and configure it as follows:

| Setting | Value |
|---|---|
| Name | `huggingface-llm-instance` |
| Instance type | `ml.t3.medium` |
| IAM role | `SageMakerExecutionRole` |

`ml.t3.medium` is sufficient for running deployment scripts. The actual model inference will run on a separate GPU endpoint, not this notebook instance.

Click **Create notebook instance** and wait for the status to show **InService**, then open it via **JupyterLab**.

---

## Step 3: Install Dependencies

Inside JupyterLab, create a `requirements.txt` file with the following contents and run the install command in a notebook cell:

```plaintext
transformers==4.53.2
torch==2.6.0
langchain==0.3.26
langchain-community==0.0.37
sagemaker==2.219.0
boto3==1.34.112
```

```python
!pip install -r requirements.txt
```

The `!` prefix is required to run shell commands from within a Jupyter cell.

---

## Step 4: Deploy the Model to a SageMaker Endpoint

The following script uses the SageMaker Python SDK to pull the [LaMini-T5-738M](https://huggingface.co/MBZUAI/LaMini-T5-738M) model from Hugging Face and deploy it to a GPU-backed endpoint.

```python
from sagemaker.huggingface import HuggingFaceModel
import sagemaker

role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"

hub = {
    'HF_MODEL_ID': 'MBZUAI/LaMini-T5-738M',
    'HF_TASK': 'text2text-generation',
}

huggingface_model = HuggingFaceModel(
    role=role,
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310",
    env=hub
)

predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    endpoint_name="lamini-t5-gpu-endpoint"
)
```

Replace `YOUR_ACCOUNT_ID` with your AWS account ID before running. Deployment typically takes 5–10 minutes. When it completes, the `predictor` object is ready to accept inference requests.

**A note on instance types:** `ml.g4dn.xlarge` is the minimum GPU instance for this model size. Attempting deployment on a CPU-only instance will succeed but inference will be impractically slow.

---

## Step 5: Retrieve the Endpoint ARN

You'll need the endpoint ARN in Part 2 to grant your Lambda function permission to invoke this endpoint.

Go to **SageMaker → Endpoints**, find `lamini-t5-gpu-endpoint`, click on it, and copy the ARN. It will follow this format:

```
arn:aws:sagemaker:us-east-2:YOUR_ACCOUNT_ID:endpoint/lamini-t5-gpu-endpoint
```

Save both the endpoint **name** and **ARN** — both are referenced in subsequent steps.

---

## Summary

At this point you have:

- An IAM execution role authorizing SageMaker to access AWS resources
- A notebook instance for running deployment and management scripts
- A live `ml.g4dn.xlarge` endpoint serving the LaMini-T5-738M model
- The endpoint ARN saved for IAM policy configuration in Part 2

**Next:** [Part 2 — Setting Up a Lambda Function for Serverless Inference](/part2_lambda_setup_guide/)
