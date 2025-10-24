"""
Test the deployed SageMaker endpoint.
"""
import json

import boto3
import numpy as np

# Configuration
ENDPOINT_NAME = "iris-classifier-endpoint"
REGION = "us-west-2"

# Create SageMaker runtime client
runtime = boto3.client("sagemaker-runtime", region_name=REGION)

# Prepare test data (iris flower features)
# [sepal_length, sepal_width, petal_length, petal_width]
test_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # Should be class 0 (setosa)
    [6.7, 3.0, 5.2, 2.3],  # Should be class 2 (virginica)
    [5.9, 3.0, 4.2, 1.5],  # Should be class 1 (versicolor)
])

print(f"Testing endpoint: {ENDPOINT_NAME}")
print(f"Region: {REGION}")
print(f"Test samples:\n{test_samples}\n")

try:
    # Invoke the endpoint
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(test_samples.tolist()),
    )

    # Parse the response
    result = json.loads(response["Body"].read().decode())
    predictions = np.array(result)

    # Display results
    print("Predictions:")
    class_names = ["setosa", "versicolor", "virginica"]
    for i, (sample, pred) in enumerate(zip(test_samples, predictions)):
        print(f"  Sample {i+1}: {sample} -> Class {pred} ({class_names[int(pred)]})")

    print("\nTest completed successfully!")

except Exception as e:
    print(f"Error invoking endpoint: {e}")
    print("\nTroubleshooting tips:")
    print("1. Verify the endpoint is in service:")
    print(f"   bentoml sagemaker status {ENDPOINT_NAME} --region {REGION}")
    print("2. Check your AWS credentials:")
    print("   aws sts get-caller-identity")
    print("3. View CloudWatch logs for errors:")
    print(f"   aws logs tail /aws/sagemaker/Endpoints/{ENDPOINT_NAME} --follow")
    raise
