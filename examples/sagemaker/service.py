"""
BentoML service for iris classification.
"""
import bentoml
import numpy as np
from bentoml.io import NumpyNdarray

# Load the trained model and create a runner
iris_clf_runner = bentoml.sklearn.get("iris_classifier:latest").to_runner()

# Create a service with the runner
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])


@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data: np.ndarray) -> np.ndarray:
    """
    Classify iris flowers based on their features.

    Args:
        input_data: Array of shape (n_samples, 4) with features:
                   [sepal_length, sepal_width, petal_length, petal_width]

    Returns:
        Array of predicted class labels (0, 1, or 2)
    """
    result = iris_clf_runner.predict.run(input_data)
    return result
