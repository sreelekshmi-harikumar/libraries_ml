# Import libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# 1. Load dataset
digits = load_digits()
df = pd.DataFrame(digits.data, columns=digits.feature_names)
df["target"] = digits.target

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

# 6. Train SVM with different kernels
results = []

for k in ["linear", "poly", "rbf"]:
    model = SVC(kernel=k)
    model.fit(Xtr, ytr)

    pred = model.predict(Xte)

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, average="macro")
    rec = recall_score(yte, pred, average="macro")
    f1 = f1_score(yte, pred, average="macro")

    print("\nKernel:", k)
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))

    results.append({
        "Kernel": k,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

# 7. Compare results using pandas
results_df = pd.DataFrame(results)

print("\nFinal Performance Comparison:")
print(results_df.round(4).to_string(index=False))

# 8. Find the best kernel by accuracy
best_kernel = results_df.loc[
    results_df["Accuracy"].idxmax(), "Kernel"
]

print("\nBest Kernel:", best_kernel)

# 9. Train the best model
best_model = SVC(kernel=best_kernel)
best_model.fit(Xtr, ytr)
best_pred = best_model.predict(Xte)

# 10. Display confusion matrix
cm = confusion_matrix(yte, best_pred)

print("\nConfusion Matrix:")
print(cm)

# 11. Display classification report
print("\nClassification Report:")
print(classification_report(yte, best_pred))
