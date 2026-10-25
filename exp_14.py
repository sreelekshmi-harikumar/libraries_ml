
# Import libraries
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
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
sc = StandardScaler().fit(Xtr)
Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

# 6. Compare activation functions
activations = ["logistic", "relu", "tanh"]
results = []

for a in activations:
    start = time.time()

    m = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation=a,
        max_iter=500,
        random_state=0
    )

    m.fit(Xtr, ytr)
    pred = m.predict(Xte)

    elapsed = time.time() - start

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, average="macro", zero_division=0)
    rec = recall_score(yte, pred, average="macro", zero_division=0)
    f1 = f1_score(yte, pred, average="macro", zero_division=0)

    print(f"\nActivation: {a}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Epochs:    {m.n_iter_}")
    print(f"Time:      {elapsed:.2f} seconds")

    results.append({
        "Activation": a,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Epochs": m.n_iter_,
        "Time (seconds)": elapsed
    })

# 7. Display comparison table
results_df = pd.DataFrame(results)

print("\nFinal Performance Comparison:")
print(results_df.round(4))

# 8. Plot accuracy comparison
plt.figure(figsize=(7, 4))
sns.barplot(data=results_df, x="Activation", y="Accuracy")
plt.title("MLP Activation Function Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()

# 9. Confusion matrix for the best activation
best_activation = results_df.loc[
    results_df["Accuracy"].idxmax(), "Activation"
]

best_model = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation=best_activation,
    max_iter=500,
    random_state=0
)
best_model.fit(Xtr, ytr)
best_pred = best_model.predict(Xte)

cm = confusion_matrix(yte, best_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {best_activation}")
plt.tight_layout()
plt.show()

print("\nBest activation:", best_activation)
print("\nClassification Report:")
print(classification_report(yte, best_pred, zero_division=0))
