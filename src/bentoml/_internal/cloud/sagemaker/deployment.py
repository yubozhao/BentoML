"""
AWS SageMaker deployment orchestration.
"""
from __future__ import annotations

import logging
import subprocess
import typing as t

from bentoml._internal.bento import Bento
from bentoml._internal.cloud.sagemaker.client import SageMakerClient
from bentoml._internal.cloud.sagemaker.config import SageMakerDeploymentConfig
from bentoml._internal.container import health_check
from bentoml.exceptions import BentoMLException

if t.TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class SageMakerDeployment:
    """
    Orchestrates the deployment of a Bento to AWS SageMaker.

    This class handles:
    1. Building and pushing Docker image to ECR
    2. Creating SageMaker model
    3. Creating endpoint configuration
    4. Creating/updating endpoint
    5. Setting up autoscaling (if enabled)
    """

    def __init__(
        self,
        config: SageMakerDeploymentConfig,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
    ):
        """
        Initialize SageMaker deployment.

        Args:
            config: Deployment configuration
            aws_access_key_id: AWS access key ID (optional)
            aws_secret_access_key: AWS secret access key (optional)
            aws_session_token: AWS session token (optional)
        """
        self.config = config
        self.client = SageMakerClient(
            region=config.region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )

    def get_execution_role(self) -> str:
        """
        Get execution role ARN, either from config or create a default one.

        Returns:
            IAM role ARN for SageMaker execution
        """
        if self.config.execution_role_arn:
            return self.config.execution_role_arn

        # Try to get default SageMaker execution role
        import boto3

        iam = boto3.client("iam", region_name=self.config.region)

        try:
            # Try to get existing SageMaker execution role
            role_name = "BentoMLSageMakerExecutionRole"
            response = iam.get_role(RoleName=role_name)
            logger.info(f"Using existing IAM role: {role_name}")
            return response["Role"]["Arn"]
        except iam.exceptions.NoSuchEntityException:
            # Create a new role
            logger.info(f"Creating IAM role: {role_name}")
            trust_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                        "Action": "sts:AssumeRole",
                    }
                ],
            }

            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=str(trust_policy),
                Description="Execution role for BentoML SageMaker deployments",
            )

            # Attach SageMaker full access policy
            iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn="arn:aws:iam::aws:policy/AmazonSageMakerFullAccess",
            )

            logger.info(f"Created IAM role: {response['Role']['Arn']}")
            return response["Role"]["Arn"]

    def build_and_push_image(self, bento: Bento) -> str:
        """
        Build Docker image for Bento and push to ECR.

        Args:
            bento: Bento to deploy

        Returns:
            ECR image URI
        """
        # Check if Docker is available
        if not health_check("docker"):
            raise BentoMLException(
                "Docker is not available. Please install Docker to build and push images."
            )

        # Get ECR repository
        repository_name = self.config.get_ecr_repository_name(bento)
        repository_uri = self.client.ensure_ecr_repository(repository_name)

        # Build image tag
        image_tag = f"{bento.tag.version}"
        image_uri = f"{repository_uri}:{image_tag}"

        logger.info(f"Building Docker image for {bento.tag}")

        # Build the Docker image using bentoml containerize
        try:
            from bentoml._internal.container import build

            # Build with the bento tag
            build(
                bento_tag=str(bento.tag),
                backend="docker",
                image_tag=[image_uri],
                progress="auto",
            )
        except Exception as e:
            raise BentoMLException(f"Failed to build Docker image: {e}") from e

        logger.info(f"Pushing image to ECR: {image_uri}")

        # Login to ECR
        ecr_password = self.client.get_ecr_login_password()
        ecr_registry = repository_uri.split("/")[0]

        try:
            subprocess.run(
                ["docker", "login", "--username", "AWS", "--password-stdin", ecr_registry],
                input=ecr_password.encode(),
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise BentoMLException(f"Failed to login to ECR: {e.stderr.decode()}") from e

        # Push image to ECR
        try:
            subprocess.run(
                ["docker", "push", image_uri],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as e:
            raise BentoMLException(f"Failed to push image to ECR: {e.stderr.decode()}") from e

        logger.info(f"Successfully pushed image to: {image_uri}")
        return image_uri

    def deploy(self, bento: Bento, wait: bool = True) -> dict[str, t.Any]:
        """
        Deploy a Bento to AWS SageMaker.

        Args:
            bento: Bento to deploy
            wait: Whether to wait for endpoint to be in service

        Returns:
            Deployment information including endpoint URL
        """
        logger.info(f"Starting SageMaker deployment for {bento.tag}")

        # Step 1: Build and push Docker image
        image_uri = self.build_and_push_image(bento)

        # Step 2: Get execution role
        execution_role_arn = self.get_execution_role()

        # Step 3: Create SageMaker model
        model_name = self.config.get_model_name()
        model_arn = self.client.create_model(
            model_name=model_name,
            image_uri=image_uri,
            execution_role_arn=execution_role_arn,
            environment=self.config.environment,
            vpc_config=self.config.vpc_config,
            tags=self.config.get_tags_list(),
        )

        # Step 4: Create endpoint configuration
        endpoint_config_name = self.config.get_endpoint_config_name()
        endpoint_config_arn = self.client.create_endpoint_config(
            endpoint_config_name=endpoint_config_name,
            model_name=model_name,
            instance_type=self.config.get_instance_type(),
            instance_count=self.config.instance_count,
            data_capture_config=self.config.data_capture_config,
            tags=self.config.get_tags_list(),
            model_data_download_timeout=self.config.model_data_download_timeout,
            container_startup_health_check_timeout=self.config.container_startup_health_check_timeout,
        )

        # Step 5: Create or update endpoint
        endpoint_name = self.config.get_endpoint_name()
        endpoint_info = self.client.create_or_update_endpoint(
            endpoint_name=endpoint_name,
            endpoint_config_name=endpoint_config_name,
            tags=self.config.get_tags_list(),
            wait=wait,
        )

        # Step 6: Setup autoscaling if enabled
        if self.config.enable_autoscaling:
            logger.info("Setting up autoscaling")
            self.client.setup_autoscaling(
                endpoint_name=endpoint_name,
                variant_name="AllTraffic",
                config=self.config.autoscaling_config,
            )

        # Get endpoint URL
        endpoint_url = f"https://runtime.sagemaker.{self.config.region}.amazonaws.com/endpoints/{endpoint_name}/invocations"

        deployment_info = {
            "endpoint_name": endpoint_name,
            "endpoint_url": endpoint_url,
            "endpoint_arn": endpoint_info["EndpointArn"],
            "endpoint_status": endpoint_info["EndpointStatus"],
            "model_name": model_name,
            "model_arn": model_arn,
            "endpoint_config_name": endpoint_config_name,
            "endpoint_config_arn": endpoint_config_arn,
            "image_uri": image_uri,
            "region": self.config.region,
        }

        logger.info(f"Deployment completed successfully!")
        logger.info(f"Endpoint URL: {endpoint_url}")

        return deployment_info

    def delete(self, delete_model: bool = True) -> None:
        """
        Delete the SageMaker deployment.

        Args:
            delete_model: Whether to also delete the model
        """
        endpoint_name = self.config.get_endpoint_name()

        # Delete endpoint and endpoint config
        self.client.delete_endpoint(endpoint_name, delete_config=True)

        # Delete model if requested
        if delete_model:
            model_name = self.config.get_model_name()
            self.client.delete_model(model_name)

        logger.info(f"Deleted deployment: {endpoint_name}")

    def get_status(self) -> dict[str, t.Any]:
        """
        Get the current status of the deployment.

        Returns:
            Deployment status information
        """
        endpoint_name = self.config.get_endpoint_name()
        endpoint_info = self.client.get_endpoint_info(endpoint_name)

        endpoint_url = f"https://runtime.sagemaker.{self.config.region}.amazonaws.com/endpoints/{endpoint_name}/invocations"

        return {
            "endpoint_name": endpoint_name,
            "endpoint_url": endpoint_url,
            "endpoint_arn": endpoint_info["EndpointArn"],
            "endpoint_status": endpoint_info["EndpointStatus"],
            "creation_time": endpoint_info["CreationTime"],
            "last_modified_time": endpoint_info["LastModifiedTime"],
        }


def deploy_to_sagemaker(
    bento: str | Bento,
    config: SageMakerDeploymentConfig,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_session_token: str | None = None,
    wait: bool = True,
) -> dict[str, t.Any]:
    """
    Deploy a Bento to AWS SageMaker.

    Args:
        bento: Bento tag or Bento object to deploy
        config: Deployment configuration
        aws_access_key_id: AWS access key ID (optional)
        aws_secret_access_key: AWS secret access key (optional)
        aws_session_token: AWS session token (optional)
        wait: Whether to wait for endpoint to be in service

    Returns:
        Deployment information

    Example:
        >>> import bentoml
        >>> from bentoml._internal.cloud.sagemaker import deploy_to_sagemaker, SageMakerDeploymentConfig
        >>>
        >>> config = SageMakerDeploymentConfig(
        ...     deployment_name="my-model",
        ...     instance_type="ml.m5.xlarge",
        ...     instance_count=2,
        ...     region="us-west-2",
        ... )
        >>>
        >>> deployment_info = deploy_to_sagemaker(
        ...     bento="my_service:latest",
        ...     config=config,
        ... )
        >>> print(deployment_info["endpoint_url"])
    """
    from bentoml import get as get_bento

    if isinstance(bento, str):
        bento = get_bento(bento)

    deployment = SageMakerDeployment(
        config=config,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )

    return deployment.deploy(bento, wait=wait)
