# AWS SageMaker Deployment Example

This example demonstrates how to deploy a BentoML service to AWS SageMaker.

## Overview

This example shows how to:
- Train a simple iris classification model
- Create a BentoML service
- Deploy to AWS SageMaker using CLI or Python API
- Invoke the deployed endpoint

## Prerequisites

1. AWS account with SageMaker permissions
2. AWS CLI configured with credentials (`aws configure`)
3. Docker installed and running
4. Python 3.8+

## Installation

```bash
pip install bentoml scikit-learn boto3
```

## Files

- `train.py` - Train and save the model
- `service.py` - Define the BentoML service
- `bentofile.yaml` - Bento build configuration
- `deploy_cli.sh` - Deploy using CLI
- `deploy_python.py` - Deploy using Python API
- `test_endpoint.py` - Test the deployed endpoint

## Quick Start

### Step 1: Train and Save the Model

```bash
python train.py
```

This will train a RandomForest classifier on the iris dataset and save it to BentoML's model store.

### Step 2: Build the Bento

```bash
bentoml build
```

This creates a deployable Bento package.

### Step 3: Deploy to SageMaker

#### Option A: Using CLI

```bash
# Edit deploy_cli.sh to set your endpoint name and region
bash deploy_cli.sh
```

#### Option B: Using Python API

```bash
python deploy_python.py
```

### Step 4: Test the Endpoint

```bash
# Edit test_endpoint.py to set your endpoint name and region
python test_endpoint.py
```

## Configuration Options

### Instance Types

- For CPU workloads: `ml.m5.large`, `ml.m5.xlarge`, `ml.c5.xlarge`
- For GPU workloads: `ml.g4dn.xlarge`, `ml.g5.xlarge`, `ml.p3.2xlarge`

### Autoscaling

Enable autoscaling to handle variable traffic:

```bash
bentoml sagemaker deploy iris_classifier:latest \
    --name iris-endpoint \
    --instance-type ml.m5.xlarge \
    --instance-count 1 \
    --autoscaling \
    --min-instances 1 \
    --max-instances 5
```

## Cleanup

To delete the deployment and avoid charges:

```bash
bentoml sagemaker delete iris-endpoint --region us-west-2
```

## Cost Considerations

- SageMaker charges by instance hour
- Minimum billing is 1 hour per instance
- Use autoscaling to minimize costs during low traffic
- Delete endpoints when not in use

Estimated costs (us-west-2):
- ml.m5.large: ~$0.115/hour
- ml.m5.xlarge: ~$0.23/hour
- ml.g4dn.xlarge: ~$0.736/hour

## Troubleshooting

### Deployment fails with "Docker not found"
- Ensure Docker is installed and running
- Run `docker ps` to verify

### Deployment fails with permission errors
- Verify AWS credentials: `aws sts get-caller-identity`
- Ensure your IAM user/role has SageMaker and ECR permissions

### Endpoint creation times out
- Check CloudWatch logs for errors
- Verify the instance type is available in your region
- Ensure your Docker image is valid

## Further Reading

- [BentoML SageMaker Documentation](https://docs.bentoml.org/en/latest/integrations/sagemaker.html)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Pricing](https://aws.amazon.com/sagemaker/pricing/)
