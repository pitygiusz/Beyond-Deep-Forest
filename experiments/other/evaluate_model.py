from deepforest import CascadeForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a complex, non-linear classification dataset
X, y = make_classification(
    n_samples=10000, 
    n_features=100, 
    n_informative=50, 
    n_redundant=20, 
    n_classes=10, 
    random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = CascadeForestClassifier(random_state=42, n_trees=50, max_layers=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
