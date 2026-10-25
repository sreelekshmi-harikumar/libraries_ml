# Import libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# 1. Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# 2. Clean dataset
df = df.dropna()
df = df.drop_duplicates()

print("\nShape after cleaning:", df.shape)

# 3. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 4. Apply Bootstrap Sampling
ids = np.arange(len(y))
acc = []

model = DecisionTreeClassifier(random_state=0)

for i in range(50):
    train_ids = resample(ids, random_state=i)
    test_ids = np.setdiff1d(ids, train_ids)

    model.fit(X.iloc[train_ids], y.iloc[train_ids])
    accuracy = model.score(X.iloc[test_ids], y.iloc[test_ids])

    acc.append(accuracy)

print("\nBootstrap Results:")
print(f"Mean Accuracy: {np.mean(acc):.4f}")
print(f"Standard Deviation: {np.std(acc):.4f}")

# 5. Apply 5-Fold Cross Validation
cv = cross_val_score(model, X, y, cv=5)

print("\n5-Fold Cross Validation Results:")
print(f"Mean Accuracy: {cv.mean():.4f}")
print(f"Standard Deviation: {cv.std():.4f}")
