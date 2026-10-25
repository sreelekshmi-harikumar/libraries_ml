
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# 1. Load dataset
wine = load_wine(as_frame=True)
df = wine.frame.copy()

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
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

# 6. Train MLP with different architectures
architectures = [(5,), (50,), (50, 30), (100, 50, 25)]
results = []

for h in architectures:
    m = MLPClassifier(
        hidden_layer_sizes=h,
        max_iter=1000,
        random_state=0
    )
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, average="macro", zero_division=0)
    rec = recall_score(yte, pred, average="macro", zero_division=0)
    f1 = f1_score(yte, pred, average="macro", zero_division=0)

    print(f"\nArchitecture: {h}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    results.append({
        "Architecture": str(h),
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

# 7. Compare performance using pandas
results_df = pd.DataFrame(results)

print("\nFinal Performance Comparison:")
print(results_df.round(4))

# 8. Plot accuracy comparison
plt.figure(figsize=(9, 5))
sns.barplot(data=results_df, x="Architecture", y="Accuracy")
plt.title("MLP Performance for Different Architectures")
plt.xlabel("Hidden Layer Architecture")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# 9. Confusion matrix for the best architecture
best_architecture = eval(
    results_df.loc[results_df["Accuracy"].idxmax(), "Architecture"]
)

best_model = MLPClassifier(
    hidden_layer_sizes=best_architecture,
    max_iter=1000,
    random_state=0
)
best_model.fit(Xtr, ytr)
best_pred = best_model.predict(Xte)

cm = confusion_matrix(yte, best_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=wine.target_names,
    yticklabels=wine.target_names
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_architecture}")
plt.tight_layout()
plt.show()

print("\nBest architecture:", best_architecture)
print("\nClassification Report:")
print(classification_report(
    yte, best_pred,
    target_names=wine.target_names,
    zero_division=0
))
