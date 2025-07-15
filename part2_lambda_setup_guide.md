# 📌 Part 2: Creating AWS Lambda Function and Role for SageMaker Endpoint Invocation (with Screenshot Guides)

## 1. Create IAM Role for Lambda

**Purpose**: This role allows the Lambda function to invoke the SageMaker endpoint.

### Steps:

- Go to the **AWS Console** → **IAM** → **Roles** → **Create Role**.
- **Trusted Entity Type**: Choose **AWS Service**.
- **Use Case**: Select **Lambda**.
- Click **Next** to **Permissions** step.

✅ **Screenshot Tip**:\


✅ Screenshot the **Create role page** → **Select Lambda as trusted entity**

---

## 2. Attach SageMaker Invoke Permission Policy

**Purpose**: Grant the Lambda function permission to invoke the SageMaker endpoint.

### Steps:

- **In the same role creation process**, click **Create policy** to open a new tab.
- Switch to the **JSON** tab and paste the following policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": [
                "arn:aws:sagemaker:us-east-2:YOUR_ACCOUNT_ID:endpoint/lamini-t5-gpu-endpoint"
            ]
        }
    ]
}
```

> ✅ **Note**: Replace the `Resource` value with the **endpoint ARN** you obtained in Part 1.

- Click **Next**, give it a name like `SageMakerInvokePolicy`, and click **Create Policy**.
- Go back to the role creation tab, **refresh** and **attach** the newly created policy.

✅ **Screenshot Tip**:\


✅ Screenshot the **JSON policy creation step** showing the custom policy\


✅ Screenshot the **attached policies** before creating the role

- Name the role `generate-text-lamini-role` and create it.
- After creation, click on the role and copy the **Role ARN** — you will need this when setting up Lambda.

✅ **Screenshot Tip**:\


✅ Screenshot the **Role ARN** after creation

---

## 3. Create AWS Lambda Function

**Purpose**: Lambda serves as the backend to accept requests and forward them to the SageMaker endpoint.

### Steps:

- Go to **AWS Console** → **Lambda** → **Create function**.
- **Author from scratch**:
  - Function name: `generate-text-lamini`
  - Runtime: `Python 3.10`
  - **Execution Role**: Choose **Use an existing role** → Select `generate-text-lamini-role`.
- Click **Create Function**.

✅ **Screenshot Tip**:\


✅ Screenshot the **Create Lambda Function screen** showing name, runtime, and role

---

## 4. Lambda Function Code

In the **Function code** section of your Lambda function, replace the default code with the following:

```python
import boto3
import json

client = boto3.client('sagemaker-runtime')

ENDPOINT_NAME = 'lamini-t5-gpu-endpoint'  # Name of your SageMaker endpoint


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

✅ **Screenshot Tip**:\


✅ Screenshot the **Function code editor** with the pasted code

- Click **Deploy** to save your Lambda function.

✅ **Screenshot Tip**:\


✅ Screenshot after clicking **Deploy**

---

✅ **Summary Checklist**:

- ✅ Created IAM role for Lambda with SageMaker invoke permissions.
- ✅ Created Lambda function using the correct IAM role.
- ✅ Configured Lambda function code to forward prompts to the SageMaker endpoint.
- ✅ Deployed Lambda function and saved Lambda ARN for API Gateway setup (Part 3).

✅ **Next Step**: Proceed to Part 3 to configure API Gateway and expose the Lambda function via a public HTTP endpoint.

