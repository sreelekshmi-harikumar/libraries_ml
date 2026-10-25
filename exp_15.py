
# Import libraries
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

# 6. Hyperparameter tuning
results = []

for lr in [0.001, 0.01]:
    for bs in [32, 128]:
        for ep in [20, 100]:

            m = MLPClassifier(
                hidden_layer_sizes=(64,),
                learning_rate_init=lr,
                batch_size=bs,
                max_iter=ep,
                random_state=0
            )

            m.fit(Xtr, ytr)
            pred = m.predict(Xte)

            acc = accuracy_score(yte, pred)
            prec = precision_score(
                yte, pred, average="macro", zero_division=0
            )
            rec = recall_score(
                yte, pred, average="macro", zero_division=0
            )
            f1 = f1_score(
                yte, pred, average="macro", zero_division=0
            )

            print(
                f"lr={lr:<6} batch={bs:<4} epochs={ep:<4} "
                f"| Accuracy: {acc:.4f}"
            )

            results.append({
                "Learning Rate": lr,
                "Batch Size": bs,
                "Epochs": ep,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1
            })

# 7. Display results
results_df = pd.DataFrame(results)

print("\nHyperparameter Comparison:")
print(results_df.round(4).to_string(index=False))

# 8. Find the best combination by accuracy
best_idx = results_df["Accuracy"].idxmax()
best = results_df.loc[best_idx]

print("\nBest Hyperparameters:")
print("Learning Rate:", best["Learning Rate"])
print("Batch Size:", int(best["Batch Size"]))
print("Epochs:", int(best["Epochs"]))
print("Best Accuracy:", round(best["Accuracy"], 4))

# 9. Plot accuracy for all combinations
results_df["Configuration"] = results_df.apply(
    lambda r: f"LR={r['Learning Rate']}, B={int(r['Batch Size'])}, "
              f"E={int(r['Epochs'])}",
    axis=1
)

plt.figure(figsize=(12, 5))
sns.barplot(
    data=results_df,
    x="Configuration",
    y="Accuracy",
    color="steelblue"
)
plt.title("MLP Hyperparameter Tuning")
plt.xlabel("Hyperparameter Combination")
plt.ylabel("Accuracy")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# 10. Train the best model and display confusion matrix
best_model = MLPClassifier(
    hidden_layer_sizes=(64,),
    learning_rate_init=float(best["Learning Rate"]),
    batch_size=int(best["Batch Size"]),
    max_iter=int(best["Epochs"]),
    random_state=0
)

best_model.fit(Xtr, ytr)
best_pred = best_model.predict(Xte)

cm = confusion_matrix(yte, best_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Best Configuration")
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(classification_report(yte, best_pred, zero_division=0))
