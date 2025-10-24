#!/bin/bash

# Deploy iris classifier to AWS SageMaker using CLI
# Edit these variables for your deployment

BENTO_TAG="iris_classifier:latest"
ENDPOINT_NAME="iris-classifier-endpoint"
REGION="us-west-2"
INSTANCE_TYPE="ml.m5.large"
INSTANCE_COUNT="1"

echo "Deploying $BENTO_TAG to AWS SageMaker..."
echo "Endpoint name: $ENDPOINT_NAME"
echo "Region: $REGION"
echo "Instance: $INSTANCE_TYPE x$INSTANCE_COUNT"
echo ""

bentoml sagemaker deploy "$BENTO_TAG" \
    --name "$ENDPOINT_NAME" \
    --region "$REGION" \
    --instance-type "$INSTANCE_TYPE" \
    --instance-count "$INSTANCE_COUNT" \
    --tag "project=iris-classification" \
    --tag "environment=demo"

echo ""
echo "Deployment complete!"
echo "To test the endpoint, run: python test_endpoint.py"
echo "To delete the endpoint, run: bentoml sagemaker delete $ENDPOINT_NAME --region $REGION"
