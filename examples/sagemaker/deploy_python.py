"""
Deploy iris classifier to AWS SageMaker using Python API.
"""
import bentoml
from bentoml._internal.cloud.sagemaker import (
    SageMakerAutoscalingConfig,
    SageMakerDeploymentConfig,
    SageMakerInstanceType,
    deploy_to_sagemaker,
)

# Configuration
BENTO_TAG = "iris_classifier:latest"
ENDPOINT_NAME = "iris-classifier-endpoint"
REGION = "us-west-2"

print(f"Deploying {BENTO_TAG} to AWS SageMaker...")

# Create deployment configuration
config = SageMakerDeploymentConfig(
    deployment_name=ENDPOINT_NAME,
    instance_type=SageMakerInstanceType.ML_M5_LARGE,
    instance_count=1,
    region=REGION,
    # Optional: Enable autoscaling
    enable_autoscaling=True,
    autoscaling_config=SageMakerAutoscalingConfig(
        min_instance_count=1,
        max_instance_count=3,
        target_invocations_per_instance=1000,
    ),
    # Optional: Add tags
    tags={
        "project": "iris-classification",
        "environment": "demo",
        "deployed-by": "bentoml",
    },
)

try:
    # Deploy to SageMaker
    deployment_info = deploy_to_sagemaker(
        bento=BENTO_TAG,
        config=config,
        wait=True,  # Wait for endpoint to be in service
    )

    print("\n" + "=" * 60)
    print("Deployment Successful!")
    print("=" * 60)
    print(f"Endpoint Name: {deployment_info['endpoint_name']}")
    print(f"Endpoint Status: {deployment_info['endpoint_status']}")
    print(f"Endpoint URL: {deployment_info['endpoint_url']}")
    print(f"Model: {deployment_info['model_name']}")
    print(f"Region: {deployment_info['region']}")
    print("\nTo test the endpoint, run: python test_endpoint.py")
    print(f"To delete: bentoml sagemaker delete {ENDPOINT_NAME} --region {REGION}")

except Exception as e:
    print(f"\nDeployment failed: {e}")
    raise
