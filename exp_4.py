# Import libraries
import numpy as np
import pandas as pd

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
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# 2. Explore dataset
print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nDataset information:")
print(df.info())

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

# 8. Train and evaluate models
results = []

for name, model in models.items():
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred)
    rec = recall_score(yte, pred)
    f1 = f1_score(yte, pred)
    auc = roc_auc_score(yte, prob)

    print("\nModel:", name)
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))

    print("\nClassification Report:")
    print(classification_report(
        yte, pred,
        target_names=data.target_names
    ))

    # Confusion matrix
    cm = confusion_matrix(yte, pred)

    print("\nConfusion Matrix:")
    print(cm)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC-AUC": auc
    })

# 9. Final comparison
results_df = pd.DataFrame(results)

print("\nFinal Model Comparison:")
print(results_df.round(4).to_string(index=False))

# 10. Find the best model
best_model = results_df.loc[
    results_df["Accuracy"].idxmax(), "Model"
]

print("\nBest Model:", best_model)
