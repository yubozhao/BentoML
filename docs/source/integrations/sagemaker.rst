=============
AWS SageMaker
=============

BentoML provides native integration with AWS SageMaker for deploying machine learning models as production-ready endpoints. This integration handles containerization, model registration, and endpoint deployment automatically.

Overview
--------

AWS SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning models at scale. BentoML's SageMaker integration allows you to:

- Deploy BentoML services as SageMaker endpoints
- Automatically build and push Docker images to Amazon ECR
- Configure autoscaling policies for your endpoints
- Manage endpoint lifecycle through CLI or Python API
- Leverage SageMaker's monitoring and logging capabilities

Prerequisites
-------------

Before deploying to SageMaker, ensure you have:

1. **AWS Account**: An active AWS account with SageMaker permissions
2. **AWS Credentials**: Configured AWS credentials (via ``aws configure`` or environment variables)
3. **Docker**: Docker installed and running locally
4. **boto3**: AWS SDK for Python installed

.. code-block:: bash

    pip install boto3

5. **IAM Permissions**: Your AWS user/role needs permissions for:
   - SageMaker (model, endpoint config, endpoint operations)
   - ECR (create repository, push images)
   - IAM (create execution role, if not provided)

Quick Start
-----------

CLI Deployment
^^^^^^^^^^^^^^

The simplest way to deploy a Bento to SageMaker is using the CLI:

.. code-block:: bash

    # Build your bento first
    bentoml build

    # Deploy to SageMaker
    bentoml sagemaker deploy my_service:latest \\
        --name my-model-endpoint \\
        --region us-west-2 \\
        --instance-type ml.m5.xlarge \\
        --instance-count 2

This command will:

1. Build a Docker image from your Bento
2. Push the image to Amazon ECR
3. Create a SageMaker model
4. Create an endpoint configuration
5. Deploy the endpoint

Python API Deployment
^^^^^^^^^^^^^^^^^^^^^

For programmatic deployment, use the Python API:

.. code-block:: python

    import bentoml
    from bentoml._internal.cloud.sagemaker import (
        deploy_to_sagemaker,
        SageMakerDeploymentConfig,
        SageMakerInstanceType,
    )

    # Configure deployment
    config = SageMakerDeploymentConfig(
        deployment_name="my-model-endpoint",
        instance_type=SageMakerInstanceType.ML_M5_XLARGE,
        instance_count=2,
        region="us-west-2",
        tags={"team": "ml-platform", "environment": "production"},
    )

    # Deploy
    deployment_info = deploy_to_sagemaker(
        bento="my_service:latest",
        config=config,
    )

    print(f"Endpoint URL: {deployment_info['endpoint_url']}")

Configuration Options
---------------------

Instance Types
^^^^^^^^^^^^^^

SageMaker supports various instance types optimized for different workloads:

**CPU Instances:**

- ``ml.t2.medium``, ``ml.t2.large``, ``ml.t2.xlarge`` - General purpose, burstable
- ``ml.m5.large`` to ``ml.m5.24xlarge`` - General purpose, balanced compute/memory
- ``ml.c5.large`` to ``ml.c5.18xlarge`` - Compute optimized

**GPU Instances:**

- ``ml.p2.xlarge`` to ``ml.p2.16xlarge`` - GPU instances (K80)
- ``ml.p3.2xlarge`` to ``ml.p3.16xlarge`` - GPU instances (V100)
- ``ml.g4dn.xlarge`` to ``ml.g4dn.16xlarge`` - GPU instances (T4)
- ``ml.g5.xlarge`` to ``ml.g5.48xlarge`` - GPU instances (A10G)

.. code-block:: python

    from bentoml._internal.cloud.sagemaker import SageMakerInstanceType

    config = SageMakerDeploymentConfig(
        deployment_name="gpu-model",
        instance_type=SageMakerInstanceType.ML_G5_XLARGE,  # GPU instance
        instance_count=1,
        region="us-west-2",
    )

Autoscaling
^^^^^^^^^^^

Enable autoscaling to automatically adjust the number of instances based on traffic:

.. code-block:: python

    from bentoml._internal.cloud.sagemaker import (
        SageMakerDeploymentConfig,
        SageMakerAutoscalingConfig,
    )

    autoscaling_config = SageMakerAutoscalingConfig(
        min_instance_count=1,
        max_instance_count=10,
        target_invocations_per_instance=1000,  # Target requests per instance
        scale_in_cooldown=300,   # Cooldown before scaling in (seconds)
        scale_out_cooldown=60,   # Cooldown before scaling out (seconds)
    )

    config = SageMakerDeploymentConfig(
        deployment_name="autoscaling-endpoint",
        instance_type="ml.m5.xlarge",
        instance_count=2,  # Initial count
        region="us-west-2",
        enable_autoscaling=True,
        autoscaling_config=autoscaling_config,
    )

Or via CLI:

.. code-block:: bash

    bentoml sagemaker deploy my_service:latest \\
        --name autoscaling-endpoint \\
        --instance-type ml.m5.xlarge \\
        --instance-count 2 \\
        --autoscaling \\
        --min-instances 1 \\
        --max-instances 10

Data Capture
^^^^^^^^^^^^

Enable data capture to log inference requests and responses for monitoring:

.. code-block:: python

    from bentoml._internal.cloud.sagemaker import (
        SageMakerDeploymentConfig,
        SageMakerDataCaptureConfig,
    )

    data_capture_config = SageMakerDataCaptureConfig(
        enable=True,
        initial_sampling_percentage=100,  # Capture 100% of requests
        destination_s3_uri="s3://my-bucket/data-capture/",
        capture_modes=["Input", "Output"],  # Capture both inputs and outputs
    )

    config = SageMakerDeploymentConfig(
        deployment_name="monitored-endpoint",
        instance_type="ml.m5.xlarge",
        region="us-west-2",
        data_capture_config=data_capture_config,
    )

VPC Configuration
^^^^^^^^^^^^^^^^^

Deploy endpoints within a VPC for enhanced security:

.. code-block:: python

    config = SageMakerDeploymentConfig(
        deployment_name="vpc-endpoint",
        instance_type="ml.m5.xlarge",
        region="us-west-2",
        vpc_config={
            "SecurityGroupIds": ["sg-12345678"],
            "Subnets": ["subnet-12345678", "subnet-87654321"],
        },
    )

Environment Variables
^^^^^^^^^^^^^^^^^^^^^

Pass environment variables to your endpoint containers:

.. code-block:: python

    config = SageMakerDeploymentConfig(
        deployment_name="env-endpoint",
        instance_type="ml.m5.xlarge",
        region="us-west-2",
        environment={
            "LOG_LEVEL": "INFO",
            "MODEL_CACHE_DIR": "/tmp/model_cache",
            "WORKERS": "4",
        },
    )

Custom IAM Role
^^^^^^^^^^^^^^^

Specify a custom IAM execution role:

.. code-block:: python

    config = SageMakerDeploymentConfig(
        deployment_name="custom-role-endpoint",
        instance_type="ml.m5.xlarge",
        region="us-west-2",
        execution_role_arn="arn:aws:iam::123456789012:role/CustomSageMakerRole",
    )

If not provided, BentoML will create a default execution role with necessary permissions.

Managing Deployments
--------------------

List Endpoints
^^^^^^^^^^^^^^

View all SageMaker endpoints:

.. code-block:: bash

    bentoml sagemaker list --region us-west-2

    # Filter by name
    bentoml sagemaker list --name-contains my-model --region us-west-2

Check Deployment Status
^^^^^^^^^^^^^^^^^^^^^^^

Get detailed status of a deployment:

.. code-block:: bash

    bentoml sagemaker status my-model-endpoint --region us-west-2

Delete Deployment
^^^^^^^^^^^^^^^^^

Remove a SageMaker endpoint:

.. code-block:: bash

    # Delete endpoint and model
    bentoml sagemaker delete my-model-endpoint --region us-west-2

    # Delete endpoint but keep model
    bentoml sagemaker delete my-model-endpoint --region us-west-2 --keep-model

Invoking Endpoints
------------------

Using AWS CLI
^^^^^^^^^^^^^

After deployment, invoke your endpoint using the AWS CLI:

.. code-block:: bash

    aws sagemaker-runtime invoke-endpoint \\
        --endpoint-name my-model-endpoint \\
        --region us-west-2 \\
        --body '{"input_data": [1, 2, 3, 4, 5]}' \\
        --content-type application/json \\
        output.json

    cat output.json

Using boto3
^^^^^^^^^^^

Invoke endpoints programmatically with boto3:

.. code-block:: python

    import boto3
    import json

    runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")

    response = runtime.invoke_endpoint(
        EndpointName="my-model-endpoint",
        ContentType="application/json",
        Body=json.dumps({"input_data": [1, 2, 3, 4, 5]}),
    )

    result = json.loads(response["Body"].read().decode())
    print(result)

Using requests
^^^^^^^^^^^^^^

For direct HTTP requests:

.. code-block:: python

    import requests
    import json
    from aws_requests_auth.aws_auth import AWSRequestsAuth

    # Setup AWS authentication
    auth = AWSRequestsAuth(
        aws_access_key="YOUR_ACCESS_KEY",
        aws_secret_access_key="YOUR_SECRET_KEY",
        aws_host="runtime.sagemaker.us-west-2.amazonaws.com",
        aws_region="us-west-2",
        aws_service="sagemaker",
    )

    endpoint_url = "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-model-endpoint/invocations"

    response = requests.post(
        endpoint_url,
        auth=auth,
        json={"input_data": [1, 2, 3, 4, 5]},
        headers={"Content-Type": "application/json"},
    )

    print(response.json())

Complete Example
----------------

Here's a complete example deploying an image classification model:

.. code-block:: python

    # train.py - Train and save model
    import bentoml
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris

    # Load data and train
    X, y = load_iris(return_X_y=True)
    clf = RandomForestClassifier()
    clf.fit(X, y)

    # Save model
    bentoml.sklearn.save_model("iris_classifier", clf)

.. code-block:: python

    # service.py - Define BentoML service
    import bentoml
    import numpy as np
    from bentoml.io import NumpyNdarray

    iris_clf_runner = bentoml.sklearn.get("iris_classifier:latest").to_runner()

    svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

    @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
    def classify(input_series: np.ndarray) -> np.ndarray:
        result = iris_clf_runner.predict.run(input_series)
        return result

.. code-block:: yaml

    # bentofile.yaml
    service: "service:svc"
    labels:
      owner: ml-team
      project: iris-classification
    include:
      - "*.py"
    python:
      packages:
        - scikit-learn
        - numpy

.. code-block:: python

    # deploy.py - Deploy to SageMaker
    import bentoml
    from bentoml._internal.cloud.sagemaker import (
        deploy_to_sagemaker,
        SageMakerDeploymentConfig,
        SageMakerAutoscalingConfig,
    )

    # Build bento
    bento = bentoml.build("bentofile.yaml")

    # Configure deployment
    config = SageMakerDeploymentConfig(
        deployment_name="iris-classifier-prod",
        instance_type="ml.m5.xlarge",
        instance_count=2,
        region="us-west-2",
        enable_autoscaling=True,
        autoscaling_config=SageMakerAutoscalingConfig(
            min_instance_count=1,
            max_instance_count=5,
        ),
        tags={
            "team": "ml-platform",
            "project": "iris-classification",
            "environment": "production",
        },
    )

    # Deploy
    deployment_info = deploy_to_sagemaker(bento=bento, config=config)

    print(f"✓ Deployment successful!")
    print(f"  Endpoint: {deployment_info['endpoint_name']}")
    print(f"  URL: {deployment_info['endpoint_url']}")

.. code-block:: python

    # test_endpoint.py - Test the deployed endpoint
    import boto3
    import json
    import numpy as np

    runtime = boto3.client("sagemaker-runtime", region_name="us-west-2")

    # Prepare test data
    test_input = np.array([[5.1, 3.5, 1.4, 0.2]])

    # Invoke endpoint
    response = runtime.invoke_endpoint(
        EndpointName="iris-classifier-prod",
        ContentType="application/json",
        Body=json.dumps(test_input.tolist()),
    )

    result = json.loads(response["Body"].read().decode())
    print(f"Prediction: {result}")

Cost Optimization
-----------------

Here are some tips to optimize SageMaker deployment costs:

1. **Right-size instances**: Start with smaller instances and scale up based on performance metrics
2. **Use autoscaling**: Automatically scale down during low traffic periods
3. **Leverage Savings Plans**: Consider SageMaker Savings Plans for predictable workloads
4. **Monitor utilization**: Use CloudWatch to track instance utilization
5. **Delete unused endpoints**: Remove endpoints that are no longer needed

Monitoring and Logging
----------------------

SageMaker automatically integrates with AWS CloudWatch for monitoring and logging.

View Logs
^^^^^^^^^

.. code-block:: bash

    # View endpoint logs
    aws logs tail /aws/sagemaker/Endpoints/my-model-endpoint --follow

Metrics
^^^^^^^

Key metrics available in CloudWatch:

- ``Invocations``: Number of endpoint invocations
- ``ModelLatency``: Time taken by the model to respond
- ``OverheadLatency``: Time taken for overhead processing
- ``Invocation4XXErrors``: Number of 4XX errors
- ``Invocation5XXErrors``: Number of 5XX errors
- ``CPUUtilization``: CPU usage percentage
- ``MemoryUtilization``: Memory usage percentage

Troubleshooting
---------------

Deployment Fails
^^^^^^^^^^^^^^^^

If deployment fails, check:

1. **AWS credentials**: Ensure credentials are properly configured
2. **IAM permissions**: Verify you have necessary SageMaker and ECR permissions
3. **Docker**: Confirm Docker is running and accessible
4. **Region**: Ensure the region supports your chosen instance type
5. **Logs**: Check CloudWatch logs for error details

Endpoint Not Responding
^^^^^^^^^^^^^^^^^^^^^^^

If the endpoint is not responding:

1. Check endpoint status: ``bentoml sagemaker status <endpoint-name>``
2. Review CloudWatch logs for errors
3. Verify the endpoint is ``InService``
4. Check network/VPC configuration if using VPC mode
5. Ensure request format matches your service's input specification

High Latency
^^^^^^^^^^^^

To reduce latency:

1. Use a larger instance type with more compute power
2. Optimize your model inference code
3. Enable request batching in your BentoML service
4. Consider deploying to a region closer to your users
5. Use GPU instances for compute-intensive models

Further Reading
---------------

- `AWS SageMaker Documentation <https://docs.aws.amazon.com/sagemaker/>`_
- `SageMaker Pricing <https://aws.amazon.com/sagemaker/pricing/>`_
- `BentoML Containerization Guide <https://docs.bentoml.org/en/latest/concepts/deploy.html#containerize-bentos>`_
- `AWS IAM Best Practices <https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html>`_
