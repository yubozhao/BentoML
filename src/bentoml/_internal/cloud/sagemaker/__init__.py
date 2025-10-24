"""
AWS SageMaker deployment support for BentoML.

This module provides native integration with AWS SageMaker for deploying
BentoML services as SageMaker endpoints.

Example:
    >>> import bentoml
    >>> from bentoml._internal.cloud.sagemaker import (
    ...     deploy_to_sagemaker,
    ...     SageMakerDeploymentConfig,
    ...     SageMakerInstanceType,
    ... )
    >>>
    >>> config = SageMakerDeploymentConfig(
    ...     deployment_name="my-model-endpoint",
    ...     instance_type=SageMakerInstanceType.ML_M5_XLARGE,
    ...     instance_count=2,
    ...     region="us-west-2",
    ...     enable_autoscaling=True,
    ... )
    >>>
    >>> deployment_info = deploy_to_sagemaker(
    ...     bento="my_service:latest",
    ...     config=config,
    ... )
    >>> print(f"Endpoint URL: {deployment_info['endpoint_url']}")
"""

from bentoml._internal.cloud.sagemaker.client import SageMakerClient
from bentoml._internal.cloud.sagemaker.config import (
    SageMakerAutoscalingConfig,
    SageMakerDataCaptureConfig,
    SageMakerDeploymentConfig,
    SageMakerInstanceType,
)
from bentoml._internal.cloud.sagemaker.deployment import (
    SageMakerDeployment,
    deploy_to_sagemaker,
)

__all__ = [
    "SageMakerClient",
    "SageMakerDeploymentConfig",
    "SageMakerInstanceType",
    "SageMakerAutoscalingConfig",
    "SageMakerDataCaptureConfig",
    "SageMakerDeployment",
    "deploy_to_sagemaker",
]
