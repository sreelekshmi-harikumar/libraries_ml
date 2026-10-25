
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    classification_report
)

# 1. Load dataset
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

# 2. Explore dataset
print("First 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

# 3. Handle missing values and duplicates
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()
df = df.drop_duplicates()

print("\nShape after cleaning:", df.shape)

# 4. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 5. Train-test split
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Standardize features
scaler = StandardScaler()

Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

# 7. Define models
# MLE: unregularized maximum-likelihood estimate
# MAP-L1: L1 regularization (Laplace-prior interpretation)
# MAP-L2: L2 regularization (Gaussian-prior interpretation)

models = {
    "MLE": LogisticRegression(
        penalty=None, max_iter=5000
    ),
    "MAP-L1": LogisticRegression(
        penalty="l1", C=1.0, solver="liblinear", max_iter=5000
    ),
    "MAP-L2": LogisticRegression(
        penalty="l2", C=1.0, solver="lbfgs", max_iter=5000
    )
}

# 8. Train, predict and evaluate
results = []

for name, model in models.items():
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, zero_division=0)
    rec = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)
    auc = roc_auc_score(yte, prob)

    print(f"\n{name}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        yte, pred,
        target_names=data.target_names,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(yte, pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=data.target_names,
        yticklabels=data.target_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

# 9. Final comparison using pandas
results_df = pd.DataFrame(results)

print("\nFinal Model Comparison:")
print(results_df.round(4))

# 10. Plot accuracy comparison
plt.figure(figsize=(7, 4))
sns.barplot(data=results_df, x="Model", y="Accuracy")
plt.title("Logistic Regression: Accuracy Comparison")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
