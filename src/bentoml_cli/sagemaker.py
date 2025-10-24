"""
CLI commands for AWS SageMaker deployment.
"""
from __future__ import annotations

import logging
import typing as t

import click

if t.TYPE_CHECKING:
    from click import Group

logger = logging.getLogger("bentoml")


def add_sagemaker_commands(cli: Group) -> None:
    """Add SageMaker deployment commands to CLI."""

    @cli.group(name="sagemaker")
    def sagemaker_group():
        """AWS SageMaker deployment commands."""
        pass

    @sagemaker_group.command(name="deploy")
    @click.argument("bento_tag", type=click.STRING)
    @click.option(
        "-n",
        "--name",
        "deployment_name",
        type=click.STRING,
        required=True,
        help="Name for the SageMaker endpoint",
    )
    @click.option(
        "-r",
        "--region",
        type=click.STRING,
        default="us-west-2",
        help="AWS region (default: us-west-2)",
    )
    @click.option(
        "-i",
        "--instance-type",
        type=click.STRING,
        default="ml.m5.large",
        help="SageMaker instance type (default: ml.m5.large)",
    )
    @click.option(
        "-c",
        "--instance-count",
        type=click.INT,
        default=1,
        help="Initial number of instances (default: 1)",
    )
    @click.option(
        "--role-arn",
        type=click.STRING,
        default=None,
        help="IAM role ARN for SageMaker execution (optional, will create if not provided)",
    )
    @click.option(
        "--autoscaling/--no-autoscaling",
        default=False,
        help="Enable autoscaling (default: disabled)",
    )
    @click.option(
        "--min-instances",
        type=click.INT,
        default=1,
        help="Minimum number of instances for autoscaling (default: 1)",
    )
    @click.option(
        "--max-instances",
        type=click.INT,
        default=10,
        help="Maximum number of instances for autoscaling (default: 10)",
    )
    @click.option(
        "--no-wait",
        is_flag=True,
        default=False,
        help="Don't wait for endpoint to be in service",
    )
    @click.option(
        "--tag",
        "tags",
        multiple=True,
        type=click.STRING,
        help="Tags for SageMaker resources (format: KEY=VALUE)",
    )
    def deploy(
        bento_tag: str,
        deployment_name: str,
        region: str,
        instance_type: str,
        instance_count: int,
        role_arn: str | None,
        autoscaling: bool,
        min_instances: int,
        max_instances: int,
        no_wait: bool,
        tags: tuple[str, ...],
    ):
        """Deploy a Bento to AWS SageMaker.

        \b
        Example:
            bentoml sagemaker deploy my_service:latest \\
                --name my-model-endpoint \\
                --region us-west-2 \\
                --instance-type ml.m5.xlarge \\
                --instance-count 2 \\
                --autoscaling \\
                --max-instances 5
        """
        try:
            from bentoml._internal.cloud.sagemaker import (
                SageMakerAutoscalingConfig,
                SageMakerDeploymentConfig,
                deploy_to_sagemaker,
            )
            from bentoml.exceptions import BentoMLException
        except ImportError as e:
            raise BentoMLException(
                "Failed to import SageMaker deployment module. "
                "Make sure boto3 is installed: pip install boto3"
            ) from e

        # Parse tags
        parsed_tags = {}
        for tag in tags:
            if "=" not in tag:
                logger.warning(f"Invalid tag format (expected KEY=VALUE): {tag}")
                continue
            key, value = tag.split("=", 1)
            parsed_tags[key] = value

        # Build autoscaling config
        autoscaling_config = SageMakerAutoscalingConfig(
            min_instance_count=min_instances,
            max_instance_count=max_instances,
        )

        # Build deployment config
        config = SageMakerDeploymentConfig(
            deployment_name=deployment_name,
            instance_type=instance_type,
            instance_count=instance_count,
            region=region,
            execution_role_arn=role_arn,
            enable_autoscaling=autoscaling,
            autoscaling_config=autoscaling_config,
            tags=parsed_tags,
        )

        logger.info(f"Deploying {bento_tag} to AWS SageMaker as {deployment_name}")
        logger.info(f"Instance: {instance_type} x{instance_count}")
        logger.info(f"Region: {region}")
        if autoscaling:
            logger.info(f"Autoscaling: enabled (min={min_instances}, max={max_instances})")

        try:
            deployment_info = deploy_to_sagemaker(
                bento=bento_tag,
                config=config,
                wait=not no_wait,
            )

            click.echo("\n" + "=" * 60)
            click.echo("Deployment Successful!")
            click.echo("=" * 60)
            click.echo(f"Endpoint Name: {deployment_info['endpoint_name']}")
            click.echo(f"Endpoint Status: {deployment_info['endpoint_status']}")
            click.echo(f"Endpoint URL: {deployment_info['endpoint_url']}")
            click.echo(f"Model: {deployment_info['model_name']}")
            click.echo(f"Region: {deployment_info['region']}")
            click.echo("\nTo invoke the endpoint:")
            click.echo(f"  aws sagemaker-runtime invoke-endpoint \\")
            click.echo(f"    --endpoint-name {deployment_info['endpoint_name']} \\")
            click.echo(f"    --region {deployment_info['region']} \\")
            click.echo(f"    --body '<your-json-payload>' \\")
            click.echo(f"    output.json")

        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            raise SystemExit(1)

    @sagemaker_group.command(name="delete")
    @click.argument("deployment_name", type=click.STRING)
    @click.option(
        "-r",
        "--region",
        type=click.STRING,
        default="us-west-2",
        help="AWS region (default: us-west-2)",
    )
    @click.option(
        "--keep-model",
        is_flag=True,
        default=False,
        help="Keep the SageMaker model after deleting endpoint",
    )
    def delete(deployment_name: str, region: str, keep_model: bool):
        """Delete a SageMaker deployment.

        \b
        Example:
            bentoml sagemaker delete my-model-endpoint --region us-west-2
        """
        try:
            from bentoml._internal.cloud.sagemaker import (
                SageMakerDeployment,
                SageMakerDeploymentConfig,
            )
        except ImportError as e:
            from bentoml.exceptions import BentoMLException

            raise BentoMLException(
                "Failed to import SageMaker deployment module. "
                "Make sure boto3 is installed: pip install boto3"
            ) from e

        config = SageMakerDeploymentConfig(
            deployment_name=deployment_name, region=region
        )
        deployment = SageMakerDeployment(config=config)

        logger.info(f"Deleting SageMaker deployment: {deployment_name}")
        try:
            deployment.delete(delete_model=not keep_model)
            click.echo(f"Successfully deleted deployment: {deployment_name}")
        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            raise SystemExit(1)

    @sagemaker_group.command(name="list")
    @click.option(
        "-r",
        "--region",
        type=click.STRING,
        default="us-west-2",
        help="AWS region (default: us-west-2)",
    )
    @click.option(
        "-n",
        "--name-contains",
        type=click.STRING,
        default=None,
        help="Filter endpoints by name",
    )
    @click.option(
        "--max-results",
        type=click.INT,
        default=100,
        help="Maximum number of results (default: 100)",
    )
    def list_endpoints(region: str, name_contains: str | None, max_results: int):
        """List SageMaker endpoints.

        \b
        Example:
            bentoml sagemaker list --region us-west-2
        """
        try:
            from bentoml._internal.cloud.sagemaker import SageMakerClient
        except ImportError as e:
            from bentoml.exceptions import BentoMLException

            raise BentoMLException(
                "Failed to import SageMaker deployment module. "
                "Make sure boto3 is installed: pip install boto3"
            ) from e

        client = SageMakerClient(region=region)

        try:
            endpoints = client.list_endpoints(
                name_contains=name_contains, max_results=max_results
            )

            if not endpoints:
                click.echo("No endpoints found.")
                return

            click.echo(f"\nFound {len(endpoints)} endpoint(s):\n")
            click.echo(f"{'Name':<40} {'Status':<15} {'Created':<25}")
            click.echo("-" * 80)

            for endpoint in endpoints:
                name = endpoint["EndpointName"]
                status = endpoint["EndpointStatus"]
                created = endpoint["CreationTime"].strftime("%Y-%m-%d %H:%M:%S")
                click.echo(f"{name:<40} {status:<15} {created:<25}")

        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            raise SystemExit(1)

    @sagemaker_group.command(name="status")
    @click.argument("deployment_name", type=click.STRING)
    @click.option(
        "-r",
        "--region",
        type=click.STRING,
        default="us-west-2",
        help="AWS region (default: us-west-2)",
    )
    def status(deployment_name: str, region: str):
        """Get status of a SageMaker deployment.

        \b
        Example:
            bentoml sagemaker status my-model-endpoint --region us-west-2
        """
        try:
            from bentoml._internal.cloud.sagemaker import (
                SageMakerDeployment,
                SageMakerDeploymentConfig,
            )
        except ImportError as e:
            from bentoml.exceptions import BentoMLException

            raise BentoMLException(
                "Failed to import SageMaker deployment module. "
                "Make sure boto3 is installed: pip install boto3"
            ) from e

        config = SageMakerDeploymentConfig(
            deployment_name=deployment_name, region=region
        )
        deployment = SageMakerDeployment(config=config)

        try:
            status_info = deployment.get_status()

            click.echo("\n" + "=" * 60)
            click.echo(f"Deployment: {status_info['endpoint_name']}")
            click.echo("=" * 60)
            click.echo(f"Status: {status_info['endpoint_status']}")
            click.echo(f"Endpoint URL: {status_info['endpoint_url']}")
            click.echo(f"Created: {status_info['creation_time']}")
            click.echo(f"Last Modified: {status_info['last_modified_time']}")

        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            raise SystemExit(1)
