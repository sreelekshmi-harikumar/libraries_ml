# Import libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# 1. Load dataset
wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

# 2. Clean dataset
df = df.dropna()
df = df.drop_duplicates()

print("\nShape after cleaning:", df.shape)

# 3. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 4. Train-test split
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)

# 5. Standardize features
sc = StandardScaler()
Xtr = sc.fit_transform(Xtr)
Xte = sc.transform(Xte)

# 6. Train MLP with different architectures
architectures = [(5,), (50,), (50, 30), (100, 50, 25)]
results = []

for h in architectures:
    model = MLPClassifier(
        hidden_layer_sizes=h,
        max_iter=1000,
        random_state=0
    )

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, average="macro")
    rec = recall_score(yte, pred, average="macro")
    f1 = f1_score(yte, pred, average="macro")

    print("\nArchitecture:", h)
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))

    results.append({
        "Architecture": str(h),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

# 7. Compare performance
results_df = pd.DataFrame(results)

print("\nFinal Performance Comparison:")
print(results_df.round(4).to_string(index=False))

# 8. Find the best architecture
best_idx = results_df["Accuracy"].idxmax()
best_architecture = architectures[best_idx]

print("\nBest Architecture:", best_architecture)
print("Best Accuracy:", round(results_df.loc[best_idx, "Accuracy"], 4))

# 9. Train the best model
best_model = MLPClassifier(
    hidden_layer_sizes=best_architecture,
    max_iter=1000,
    random_state=0
)

best_model.fit(Xtr, ytr)
best_pred = best_model.predict(Xte)

# 10. Display confusion matrix
cm = confusion_matrix(yte, best_pred)

print("\nConfusion Matrix:")
print(cm)

# 11. Display classification report
print("\nClassification Report:")
print(classification_report(
    yte, best_pred,
    target_names=wine.target_names
))
