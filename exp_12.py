
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# 1. Load dataset
digits = load_digits(as_frame=True)
df = digits.frame.copy()

print("First 5 rows:")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nDataset information:")
df.info()

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
    m = SVC(kernel=k)
    m.fit(Xtr, ytr)

    pred = m.predict(Xte)

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, average="macro", zero_division=0)
    rec = recall_score(yte, pred, average="macro", zero_division=0)
    f1 = f1_score(yte, pred, average="macro", zero_division=0)

    print(f"\nKernel: {k}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(yte, pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {k} kernel")
    plt.show()

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
print(results_df.round(4))

# 8. Plot accuracy comparison
plt.figure(figsize=(7, 4))
sns.barplot(data=results_df, x="Kernel", y="Accuracy")
plt.title("SVM Kernel Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# 9. Classification report for the best kernel by accuracy
best_kernel = results_df.loc[
    results_df["Accuracy"].idxmax(), "Kernel"
]

best_model = SVC(kernel=best_kernel)
best_model.fit(Xtr, ytr)
best_pred = best_model.predict(Xte)

print("\nBest kernel:", best_kernel)
print("\nClassification Report:")
print(classification_report(yte, best_pred, zero_division=0))
