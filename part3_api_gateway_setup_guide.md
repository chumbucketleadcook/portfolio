---
layout: post
title: "How to Deploy a Large Language Model from Hugging Face to AWS: Part 3 — API Gateway Setup"
date: 2025-10-01
category: ml
---
# 📌 Part 3: Creating an HTTP API in AWS API Gateway to Access Lambda (with Screenshot Guides)

## 1. Create an HTTP API via API Gateway

**Purpose**: The HTTP API allows you to expose the Lambda function via a public HTTP endpoint that you can call from external applications (e.g., Postman, Jupyter Notebook).

### Steps:

- Go to the **AWS Console** → **API Gateway** → **APIs**.
- Click on **Create API**.
- Choose **HTTP API** and click **Build**.

🔹 **Screenshot Tip**:\
🔹 Screenshot of the **API Gateway landing page** → **Create API**

---

## 2. Configure API Integration with Lambda

### Steps:

- **Add Integration**:
  - Integration type: **Lambda function**.
  - Choose your Lambda function: `generate-text-lamini`.
  - Check the box **Add permissions to Lambda function** — this allows API Gateway to invoke your Lambda automatically.
- Click **Next**.

🔹 **Screenshot Tip**:\
🔹 Screenshot the **Add Integration** screen showing your Lambda function selected

---

## 3. Configure Route

### Steps:

- **Define Route**:
  - Method: **POST**
  - Resource path: `/generate`
- Click **Next**.

🔹 **Screenshot Tip**:\
🔹 Screenshot the **Route configuration screen** showing `POST` and `/generate`

---

## 4. Configure Stage

### Steps:

- Stage name: leave as **\$default** (or set a custom stage name if preferred).
- Enable auto-deploy (this publishes changes automatically without extra steps).
- Click **Next** → **Create**.

🔹 **Screenshot Tip**:\
🔹 Screenshot the **Stage configuration screen** with **\$default** and **Auto deploy** enabled

---

## 5. Find Your Invoke URL

### Steps:

- After creation, you will land on the **API dashboard**.
- You’ll see the **Invoke URL** at the top.\
  Format example:

```
https://your-api-id.execute-api.us-east-2.amazonaws.com
```

- Append `/generate` to the Invoke URL.\
  🔹 **Final API Endpoint**:

```
https://your-api-id.execute-api.us-east-2.amazonaws.com/generate
```

🔹 **Screenshot Tip**:\
🔹 Screenshot of the **API Invoke URL** and the **Routes tab** showing `/generate`

---

## 6. (Optional) Test API Directly in AWS Console

- Go to **API Gateway** → **APIs** → your API → **Routes** → **POST /generate** → **Test**.
- Request Body:

```json
{
  "prompt": "Write a poem about the ocean."
}
```

- Click **Test** and verify the Lambda/SageMaker pipeline works.

🔹 **Screenshot Tip**:\
🔹 Screenshot of the **API Gateway testing console** showing request/response

---

## 7. Example API Request from External Client (Python)

You can now interact with your model remotely using a simple Python script.

```python
import requests

# Replace with your actual invoke URL
url = "https://your-api-id.execute-api.us-east-2.amazonaws.com/generate"

data = {"prompt": "Write an article about deploying LLM models to AWS services"}

response = requests.post(url, json=data)

print(response.json())
```

🔹 **Screenshot Tip**:\
🔹 Optional: Screenshot of API call from a Jupyter Notebook or terminal

---

🔹 **Summary Checklist**:

- ✅ Created HTTP API with Lambda integration.
- ✅ Configured `POST /generate` route linked to Lambda.
- ✅ Auto-deployed to `$default` stage.
- ✅ Retrieved and validated public HTTP endpoint.
- ✅ Tested end-to-end pipeline from API Gateway → Lambda → SageMaker Endpoint.

---

🔹 **Important Notes**:

- ✅ Ensure your Lambda function’s IAM role includes `sagemaker:InvokeEndpoint` permission (covered in Part 2).
- ✅ The **Invoke URL** is what you will use to send prompts to your model.
- ✅ You can monitor Lambda/API logs in **CloudWatch Logs**.

---

🔹 **Next Step**: You are now ready to send HTTP requests to your SageMaker-powered LLM model through API Gateway!

