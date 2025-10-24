"""
Train a simple iris classification model and save it with BentoML.
"""
import bentoml
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save the model to BentoML's model store
saved_model = bentoml.sklearn.save_model("iris_classifier", clf)

print(f"Model saved: {saved_model}")
print(f"Model tag: {saved_model.tag}")
