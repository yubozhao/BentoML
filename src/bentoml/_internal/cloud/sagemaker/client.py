"""
AWS SageMaker client wrapper for deployment operations.
"""
from __future__ import annotations

import logging
import time
import typing as t

if t.TYPE_CHECKING:
    from bentoml._internal.cloud.sagemaker.config import (
        SageMakerAutoscalingConfig,
        SageMakerDataCaptureConfig,
        SageMakerDeploymentConfig,
    )

logger = logging.getLogger(__name__)


class SageMakerClient:
    """
    Client wrapper for AWS SageMaker operations.

    Handles model registration, endpoint configuration, and endpoint deployment.
    """

    def __init__(
        self,
        region: str,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
    ):
        """
        Initialize SageMaker client.

        Args:
            region: AWS region
            aws_access_key_id: AWS access key ID (optional, uses environment/IAM if not provided)
            aws_secret_access_key: AWS secret access key (optional)
            aws_session_token: AWS session token (optional)
        """
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for SageMaker deployment. "
                "Install it with: pip install boto3"
            )

        self.region = region
        session_kwargs: dict[str, t.Any] = {"region_name": region}

        if aws_access_key_id:
            session_kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            session_kwargs["aws_secret_access_key"] = aws_secret_access_key
        if aws_session_token:
            session_kwargs["aws_session_token"] = aws_session_token

        self.session = boto3.Session(**session_kwargs)
        self.sagemaker = self.session.client("sagemaker")
        self.ecr = self.session.client("ecr")
        self.sts = self.session.client("sts")
        self.autoscaling = self.session.client("application-autoscaling")

    def get_account_id(self) -> str:
        """Get AWS account ID."""
        response = self.sts.get_caller_identity()
        return response["Account"]

    def ensure_ecr_repository(self, repository_name: str) -> str:
        """
        Ensure ECR repository exists, create if it doesn't.

        Args:
            repository_name: Name of the ECR repository

        Returns:
            Repository URI
        """
        try:
            response = self.ecr.describe_repositories(
                repositoryNames=[repository_name]
            )
            repository_uri = response["repositories"][0]["repositoryUri"]
            logger.info(f"ECR repository already exists: {repository_uri}")
            return repository_uri
        except self.ecr.exceptions.RepositoryNotFoundException:
            logger.info(f"Creating ECR repository: {repository_name}")
            response = self.ecr.create_repository(
                repositoryName=repository_name,
                imageScanningConfiguration={"scanOnPush": True},
                encryptionConfiguration={"encryptionType": "AES256"},
            )
            return response["repository"]["repositoryUri"]

    def get_ecr_login_password(self) -> str:
        """Get ECR authorization token for Docker login."""
        response = self.ecr.get_authorization_token()
        import base64

        auth_data = response["authorizationData"][0]
        token = base64.b64decode(auth_data["authorizationToken"]).decode()
        # Token format is "AWS:password"
        return token.split(":")[1]

    def create_model(
        self,
        model_name: str,
        image_uri: str,
        execution_role_arn: str,
        environment: dict[str, str] | None = None,
        vpc_config: dict[str, t.Any] | None = None,
        tags: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Create a SageMaker model.

        Args:
            model_name: Name for the model
            image_uri: URI of the Docker image
            execution_role_arn: IAM role ARN for SageMaker
            environment: Environment variables
            vpc_config: VPC configuration
            tags: Resource tags

        Returns:
            Model ARN
        """
        try:
            # Check if model already exists
            self.sagemaker.describe_model(ModelName=model_name)
            logger.info(f"Model {model_name} already exists, deleting and recreating")
            self.sagemaker.delete_model(ModelName=model_name)
        except self.sagemaker.exceptions.ClientError as e:
            if "Could not find model" not in str(e):
                raise

        logger.info(f"Creating SageMaker model: {model_name}")

        primary_container: dict[str, t.Any] = {
            "Image": image_uri,
            "Mode": "SingleModel",
        }

        if environment:
            primary_container["Environment"] = environment

        create_model_kwargs: dict[str, t.Any] = {
            "ModelName": model_name,
            "PrimaryContainer": primary_container,
            "ExecutionRoleArn": execution_role_arn,
        }

        if vpc_config:
            create_model_kwargs["VpcConfig"] = vpc_config

        if tags:
            create_model_kwargs["Tags"] = tags

        response = self.sagemaker.create_model(**create_model_kwargs)
        logger.info(f"Created model: {response['ModelArn']}")
        return response["ModelArn"]

    def create_endpoint_config(
        self,
        endpoint_config_name: str,
        model_name: str,
        instance_type: str,
        instance_count: int,
        data_capture_config: SageMakerDataCaptureConfig | None = None,
        tags: list[dict[str, str]] | None = None,
        model_data_download_timeout: int = 900,
        container_startup_health_check_timeout: int = 600,
    ) -> str:
        """
        Create a SageMaker endpoint configuration.

        Args:
            endpoint_config_name: Name for the endpoint configuration
            model_name: Name of the model to deploy
            instance_type: EC2 instance type
            instance_count: Number of instances
            data_capture_config: Data capture configuration
            tags: Resource tags
            model_data_download_timeout: Timeout for model download
            container_startup_health_check_timeout: Timeout for container health check

        Returns:
            Endpoint config ARN
        """
        try:
            # Check if endpoint config already exists
            self.sagemaker.describe_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
            logger.info(
                f"Endpoint config {endpoint_config_name} already exists, deleting and recreating"
            )
            self.sagemaker.delete_endpoint_config(
                EndpointConfigName=endpoint_config_name
            )
        except self.sagemaker.exceptions.ClientError as e:
            if "Could not find endpoint configuration" not in str(e):
                raise

        logger.info(f"Creating endpoint configuration: {endpoint_config_name}")

        production_variants = [
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": instance_count,
                "InstanceType": instance_type,
                "ModelDataDownloadTimeoutInSeconds": model_data_download_timeout,
                "ContainerStartupHealthCheckTimeoutInSeconds": container_startup_health_check_timeout,
            }
        ]

        create_config_kwargs: dict[str, t.Any] = {
            "EndpointConfigName": endpoint_config_name,
            "ProductionVariants": production_variants,
        }

        if data_capture_config and data_capture_config.enable:
            if not data_capture_config.destination_s3_uri:
                raise ValueError(
                    "destination_s3_uri is required when data capture is enabled"
                )
            create_config_kwargs["DataCaptureConfig"] = {
                "EnableCapture": True,
                "InitialSamplingPercentage": data_capture_config.initial_sampling_percentage,
                "DestinationS3Uri": data_capture_config.destination_s3_uri,
                "CaptureOptions": [
                    {"CaptureMode": mode} for mode in data_capture_config.capture_modes
                ],
            }

        if tags:
            create_config_kwargs["Tags"] = tags

        response = self.sagemaker.create_endpoint_config(**create_config_kwargs)
        logger.info(f"Created endpoint config: {response['EndpointConfigArn']}")
        return response["EndpointConfigArn"]

    def create_or_update_endpoint(
        self,
        endpoint_name: str,
        endpoint_config_name: str,
        tags: list[dict[str, str]] | None = None,
        wait: bool = True,
    ) -> dict[str, t.Any]:
        """
        Create or update a SageMaker endpoint.

        Args:
            endpoint_name: Name for the endpoint
            endpoint_config_name: Name of the endpoint configuration
            tags: Resource tags
            wait: Whether to wait for endpoint to be in service

        Returns:
            Endpoint information
        """
        try:
            # Check if endpoint already exists
            endpoint_info = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint {endpoint_name} already exists, updating")
            self.sagemaker.update_endpoint(
                EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
            )
            action = "update"
        except self.sagemaker.exceptions.ClientError as e:
            if "Could not find endpoint" not in str(e):
                raise
            logger.info(f"Creating endpoint: {endpoint_name}")
            create_endpoint_kwargs: dict[str, t.Any] = {
                "EndpointName": endpoint_name,
                "EndpointConfigName": endpoint_config_name,
            }
            if tags:
                create_endpoint_kwargs["Tags"] = tags
            self.sagemaker.create_endpoint(**create_endpoint_kwargs)
            action = "create"

        if wait:
            logger.info(
                f"Waiting for endpoint {endpoint_name} to be in service (this may take several minutes)..."
            )
            self._wait_for_endpoint(endpoint_name)

        endpoint_info = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
        logger.info(
            f"Endpoint {endpoint_name} is {endpoint_info['EndpointStatus']} ({action}d)"
        )
        return endpoint_info

    def _wait_for_endpoint(
        self, endpoint_name: str, timeout: int = 1800, interval: int = 30
    ) -> None:
        """
        Wait for endpoint to be in service.

        Args:
            endpoint_name: Name of the endpoint
            timeout: Maximum time to wait in seconds
            interval: Polling interval in seconds
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
            status = response["EndpointStatus"]

            if status == "InService":
                return
            elif status in ["Failed", "RollingBack"]:
                failure_reason = response.get("FailureReason", "Unknown")
                raise RuntimeError(
                    f"Endpoint {endpoint_name} failed: {status} - {failure_reason}"
                )

            logger.info(f"Endpoint status: {status}, waiting...")
            time.sleep(interval)

        raise TimeoutError(
            f"Endpoint {endpoint_name} did not reach InService status within {timeout} seconds"
        )

    def setup_autoscaling(
        self,
        endpoint_name: str,
        variant_name: str,
        config: SageMakerAutoscalingConfig,
    ) -> None:
        """
        Setup autoscaling for a SageMaker endpoint.

        Args:
            endpoint_name: Name of the endpoint
            variant_name: Name of the variant to autoscale
            config: Autoscaling configuration
        """
        resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"

        logger.info(f"Registering scalable target for {resource_id}")
        self.autoscaling.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            MinCapacity=config.min_instance_count,
            MaxCapacity=config.max_instance_count,
        )

        logger.info(f"Setting up target tracking scaling policy for {resource_id}")
        self.autoscaling.put_scaling_policy(
            PolicyName=f"{endpoint_name}-scaling-policy",
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:variant:DesiredInstanceCount",
            PolicyType="TargetTrackingScaling",
            TargetTrackingScalingPolicyConfiguration={
                "TargetValue": float(config.target_invocations_per_instance),
                "PredefinedMetricSpecification": {
                    "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
                },
                "ScaleInCooldown": config.scale_in_cooldown,
                "ScaleOutCooldown": config.scale_out_cooldown,
            },
        )
        logger.info("Autoscaling configured successfully")

    def delete_endpoint(self, endpoint_name: str, delete_config: bool = True) -> None:
        """
        Delete a SageMaker endpoint.

        Args:
            endpoint_name: Name of the endpoint
            delete_config: Whether to also delete the endpoint configuration
        """
        try:
            logger.info(f"Deleting endpoint: {endpoint_name}")
            self.sagemaker.delete_endpoint(EndpointName=endpoint_name)

            if delete_config:
                endpoint_config_name = f"{endpoint_name}-config"
                try:
                    logger.info(f"Deleting endpoint config: {endpoint_config_name}")
                    self.sagemaker.delete_endpoint_config(
                        EndpointConfigName=endpoint_config_name
                    )
                except self.sagemaker.exceptions.ClientError as e:
                    if "Could not find endpoint configuration" not in str(e):
                        logger.warning(
                            f"Failed to delete endpoint config {endpoint_config_name}: {e}"
                        )

        except self.sagemaker.exceptions.ClientError as e:
            if "Could not find endpoint" in str(e):
                logger.info(f"Endpoint {endpoint_name} not found, nothing to delete")
            else:
                raise

    def delete_model(self, model_name: str) -> None:
        """
        Delete a SageMaker model.

        Args:
            model_name: Name of the model
        """
        try:
            logger.info(f"Deleting model: {model_name}")
            self.sagemaker.delete_model(ModelName=model_name)
        except self.sagemaker.exceptions.ClientError as e:
            if "Could not find model" in str(e):
                logger.info(f"Model {model_name} not found, nothing to delete")
            else:
                raise

    def list_endpoints(
        self, name_contains: str | None = None, max_results: int = 100
    ) -> list[dict[str, t.Any]]:
        """
        List SageMaker endpoints.

        Args:
            name_contains: Filter by endpoint name
            max_results: Maximum number of results

        Returns:
            List of endpoint summaries
        """
        list_kwargs: dict[str, t.Any] = {
            "MaxResults": max_results,
            "SortBy": "CreationTime",
            "SortOrder": "Descending",
        }

        if name_contains:
            list_kwargs["NameContains"] = name_contains

        response = self.sagemaker.list_endpoints(**list_kwargs)
        return response["Endpoints"]

    def get_endpoint_info(self, endpoint_name: str) -> dict[str, t.Any]:
        """
        Get detailed information about an endpoint.

        Args:
            endpoint_name: Name of the endpoint

        Returns:
            Endpoint information
        """
        return self.sagemaker.describe_endpoint(EndpointName=endpoint_name)
