# 📌 Part 1: Deploying a Hugging Face Model to SageMaker Endpoint (with Screenshot Guides)

## 1. Create an IAM Execution Role

**Purpose**: This role allows SageMaker to access your AWS resources (like ECR, S3, and other services).

### Steps:

- Go to the **AWS Console** → **IAM** → **Roles** → **Create role**.
- **Trusted Entity**: Choose **SageMaker**.
- **Permissions policies**: Attach the following managed policies:
  - `AmazonSageMakerFullAccess`
  - `AmazonS3FullAccess`
- **Role Name**: Name it something like `SageMakerExecutionRole`.
- After creation, click on the role and copy the **Role ARN** — you'll need this when deploying the model.

**📷 Screenshot Tip**:\
✅ Screenshot the **Create role page** → **Select trusted entity as SageMaker**\
✅ Screenshot the **Attach permissions policies** step showing both attached policies\
✅ Screenshot the **Role ARN** after creation

---

## 2. Launch a SageMaker Notebook Instance

**Purpose**: This notebook instance will be used to run Python code to deploy your model.

### Steps:

- Go to **SageMaker** → **Notebook instances** → **Create notebook instance**.
- Name: `huggingface-llm-instance`.
- Instance type: `ml.t3.medium` (or larger if needed).
- **IAM Role**: Choose the `SageMakerExecutionRole` you created.
- **Git repositories**: Skip this.
- Click **Create notebook instance**.

**📷 Screenshot Tip**:\
✅ Screenshot the **Create notebook instance** screen showing the name and instance type\
✅ Screenshot the **IAM role** selection

---

## 3. Add `requirements.txt` to Your Notebook Instance

**Purpose**: To ensure consistent environments, install all necessary Python dependencies at once.

### Example `requirements.txt`:

```plaintext
transformers==4.53.2
torch==2.6.0
langchain==0.3.26
langchain-community==0.0.37
sagemaker==2.219.0
boto3==1.34.112
```

### Upload Instructions:

- Open your notebook instance via **JupyterLab**.
- Drag and drop the `requirements.txt` file into your notebook environment.

**📷 Screenshot Tip**:\
✅ Screenshot the **JupyterLab interface** with the uploaded `requirements.txt` file visible in the left sidebar

---

## 4. Install Python Packages

In a new Jupyter notebook cell, run:

```python
# Install all necessary packages listed in requirements.txt
!pip install -r requirements.txt
```

✅ **Note**: The `!` prefix is necessary to run shell commands within JupyterLab.

**📷 Screenshot Tip**:\
✅ Screenshot the notebook cell running `!pip install -r requirements.txt` and the successful installation output

---

## 5. Deploy the Hugging Face Model to SageMaker Endpoint

We'll use `sagemaker` SDK to deploy the MBZUAI LaMini T5 738M model.

```python
from sagemaker.huggingface import HuggingFaceModel
import sagemaker

# Specify your execution role ARN
role = "arn:aws:iam::YOUR_ACCOUNT_ID:role/SageMakerExecutionRole"

# Initialize the SageMaker session
sess = sagemaker.Session()

# Specify the model checkpoint from Hugging Face
hub_model_id = "MBZUAI/LaMini-T5-738M"

# Configure environment variables for the model container
hub = {
    'HF_MODEL_ID': hub_model_id,
    'HF_TASK': 'text2text-generation',
}

# Create a HuggingFaceModel object
huggingface_model = HuggingFaceModel(
    role=role,
    transformers_version="4.37.0",
    pytorch_version="2.1.0",
    py_version="py310",
    env=hub
)

# Deploy the model to an endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",  # GPU instance
    endpoint_name="lamini-t5-gpu-endpoint"
)
```

**📷 Screenshot Tip**:\
✅ Screenshot the notebook cell after the endpoint is successfully deployed showing the predictor output\
✅ Screenshot the **SageMaker Console** → **Endpoints** showing the new endpoint `lamini-t5-gpu-endpoint`

---

## 6. Finding Your Model Endpoint ARN

- Go to **SageMaker Console** → **Endpoints**.
- Find `lamini-t5-gpu-endpoint`.
- **Click** on it → Copy the **Endpoint ARN** (you will need this for Lambda permissions later).

✅ **Example ARN Format**:

```
arn:aws:sagemaker:us-east-2:YOUR_ACCOUNT_ID:endpoint/lamini-t5-gpu-endpoint
```

**📷 Screenshot Tip**:\
✅ Screenshot the **Endpoint configuration page** showing the **ARN** field

---

✅ **Summary Checklist**:

- Created IAM execution role.
- Launched SageMaker notebook instance.
- Added and installed requirements.txt.
- Deployed Hugging Face model to endpoint.
- Saved endpoint name and ARN for future use.



