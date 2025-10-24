"""
AWS SageMaker deployment configuration classes.
"""
from __future__ import annotations

import typing as t
from enum import Enum

import attr

if t.TYPE_CHECKING:
    from bentoml._internal.bento import Bento


class SageMakerInstanceType(str, Enum):
    """Common SageMaker instance types."""

    # CPU instances
    ML_T2_MEDIUM = "ml.t2.medium"
    ML_T2_LARGE = "ml.t2.large"
    ML_T2_XLARGE = "ml.t2.xlarge"
    ML_T2_2XLARGE = "ml.t2.2xlarge"
    ML_M5_LARGE = "ml.m5.large"
    ML_M5_XLARGE = "ml.m5.xlarge"
    ML_M5_2XLARGE = "ml.m5.2xlarge"
    ML_M5_4XLARGE = "ml.m5.4xlarge"
    ML_M5_12XLARGE = "ml.m5.12xlarge"
    ML_M5_24XLARGE = "ml.m5.24xlarge"
    ML_C5_LARGE = "ml.c5.large"
    ML_C5_XLARGE = "ml.c5.xlarge"
    ML_C5_2XLARGE = "ml.c5.2xlarge"
    ML_C5_4XLARGE = "ml.c5.4xlarge"
    ML_C5_9XLARGE = "ml.c5.9xlarge"
    ML_C5_18XLARGE = "ml.c5.18xlarge"

    # GPU instances
    ML_P2_XLARGE = "ml.p2.xlarge"
    ML_P2_8XLARGE = "ml.p2.8xlarge"
    ML_P2_16XLARGE = "ml.p2.16xlarge"
    ML_P3_2XLARGE = "ml.p3.2xlarge"
    ML_P3_8XLARGE = "ml.p3.8xlarge"
    ML_P3_16XLARGE = "ml.p3.16xlarge"
    ML_G4DN_XLARGE = "ml.g4dn.xlarge"
    ML_G4DN_2XLARGE = "ml.g4dn.2xlarge"
    ML_G4DN_4XLARGE = "ml.g4dn.4xlarge"
    ML_G4DN_8XLARGE = "ml.g4dn.8xlarge"
    ML_G4DN_12XLARGE = "ml.g4dn.12xlarge"
    ML_G4DN_16XLARGE = "ml.g4dn.16xlarge"
    ML_G5_XLARGE = "ml.g5.xlarge"
    ML_G5_2XLARGE = "ml.g5.2xlarge"
    ML_G5_4XLARGE = "ml.g5.4xlarge"
    ML_G5_8XLARGE = "ml.g5.8xlarge"
    ML_G5_12XLARGE = "ml.g5.12xlarge"
    ML_G5_16XLARGE = "ml.g5.16xlarge"
    ML_G5_24XLARGE = "ml.g5.24xlarge"
    ML_G5_48XLARGE = "ml.g5.48xlarge"


@attr.frozen
class SageMakerAutoscalingConfig:
    """Configuration for SageMaker endpoint autoscaling."""

    min_instance_count: int = attr.field(default=1, validator=attr.validators.ge(1))
    max_instance_count: int = attr.field(default=10, validator=attr.validators.ge(1))
    target_invocations_per_instance: int = attr.field(default=1000, validator=attr.validators.ge(1))
    scale_in_cooldown: int = attr.field(default=300, validator=attr.validators.ge(0))
    scale_out_cooldown: int = attr.field(default=60, validator=attr.validators.ge(0))

    @max_instance_count.validator
    def _validate_max_instance_count(
        self, attribute: attr.Attribute[int], value: int
    ) -> None:
        if value < self.min_instance_count:
            raise ValueError(
                f"max_instance_count ({value}) must be >= min_instance_count ({self.min_instance_count})"
            )


@attr.frozen
class SageMakerDataCaptureConfig:
    """Configuration for SageMaker data capture."""

    enable: bool = attr.field(default=False)
    initial_sampling_percentage: int = attr.field(default=100, validator=attr.validators.instance_of(int))
    destination_s3_uri: str | None = attr.field(default=None)
    capture_modes: list[str] = attr.field(factory=lambda: ["Input", "Output"])

    @initial_sampling_percentage.validator
    def _validate_sampling_percentage(
        self, attribute: attr.Attribute[int], value: int
    ) -> None:
        if not 0 <= value <= 100:
            raise ValueError(
                f"initial_sampling_percentage must be between 0 and 100, got {value}"
            )


@attr.frozen
class SageMakerDeploymentConfig:
    """
    Configuration for deploying a Bento to AWS SageMaker.

    Args:
        deployment_name: Name for the SageMaker endpoint and model
        instance_type: EC2 instance type for the endpoint
        instance_count: Initial number of instances
        region: AWS region for deployment
        execution_role_arn: IAM role ARN for SageMaker execution
        ecr_repository_name: ECR repository name for Docker image (optional)
        vpc_config: VPC configuration for the endpoint (optional)
        tags: Tags to apply to SageMaker resources
        enable_autoscaling: Whether to enable autoscaling
        autoscaling_config: Autoscaling configuration
        data_capture_config: Data capture configuration
        environment: Environment variables for the endpoint
        model_data_download_timeout: Timeout for model data download in seconds
        container_startup_health_check_timeout: Timeout for container startup health check in seconds
    """

    deployment_name: str
    instance_type: str | SageMakerInstanceType = attr.field(
        default=SageMakerInstanceType.ML_M5_LARGE
    )
    instance_count: int = attr.field(default=1, validator=attr.validators.ge(1))
    region: str = attr.field(default="us-west-2")
    execution_role_arn: str | None = attr.field(default=None)

    # Container configuration
    ecr_repository_name: str | None = attr.field(default=None)

    # Network configuration
    vpc_config: dict[str, t.Any] | None = attr.field(default=None)

    # Tags
    tags: dict[str, str] = attr.field(factory=dict)

    # Autoscaling
    enable_autoscaling: bool = attr.field(default=False)
    autoscaling_config: SageMakerAutoscalingConfig = attr.field(
        factory=SageMakerAutoscalingConfig
    )

    # Data capture
    data_capture_config: SageMakerDataCaptureConfig = attr.field(
        factory=SageMakerDataCaptureConfig
    )

    # Environment variables
    environment: dict[str, str] = attr.field(factory=dict)

    # Timeouts
    model_data_download_timeout: int = attr.field(default=900)  # 15 minutes
    container_startup_health_check_timeout: int = attr.field(default=600)  # 10 minutes

    def get_instance_type(self) -> str:
        """Get the instance type as a string."""
        if isinstance(self.instance_type, SageMakerInstanceType):
            return self.instance_type.value
        return self.instance_type

    def get_tags_list(self) -> list[dict[str, str]]:
        """Convert tags dict to SageMaker tags format."""
        return [{"Key": k, "Value": v} for k, v in self.tags.items()]

    def get_ecr_repository_name(self, bento: Bento) -> str:
        """Get the ECR repository name, using bento name as default."""
        if self.ecr_repository_name:
            return self.ecr_repository_name
        # Use bento name, replacing underscores with hyphens for ECR compatibility
        return bento.tag.name.replace("_", "-").lower()

    def get_model_name(self) -> str:
        """Get the SageMaker model name."""
        return f"{self.deployment_name}-model"

    def get_endpoint_config_name(self) -> str:
        """Get the SageMaker endpoint configuration name."""
        return f"{self.deployment_name}-config"

    def get_endpoint_name(self) -> str:
        """Get the SageMaker endpoint name."""
        return self.deployment_name

    @deployment_name.validator
    def _validate_deployment_name(
        self, attribute: attr.Attribute[str], value: str
    ) -> None:
        """Validate deployment name follows SageMaker naming conventions."""
        if not value:
            raise ValueError("deployment_name cannot be empty")
        if len(value) > 63:
            raise ValueError(f"deployment_name must be <= 63 characters, got {len(value)}")
        if not value.replace("-", "").replace("_", "").isalnum():
            raise ValueError(
                "deployment_name must contain only alphanumeric characters, hyphens, and underscores"
            )
