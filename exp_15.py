# Import libraries
import numpy as np
import pandas as pd

from sklearn.datasets import load_digits
from sklearn.neural_network import MLPClassifier
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

# 6. Hyperparameter tuning
results = []

for lr in [0.001, 0.01]:
    for bs in [32, 128]:
        for ep in [20, 100]:

            model = MLPClassifier(
                hidden_layer_sizes=(64,),
                learning_rate_init=lr,
                batch_size=bs,
                max_iter=ep,
                random_state=0
            )

            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

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

            print("Learning Rate:", lr)
            print("Batch Size:", bs)
            print("Epochs:", ep)
            print("Accuracy:", round(acc, 4))
            print()

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

# 9. Train the best model
best_model = MLPClassifier(
    hidden_layer_sizes=(64,),
    learning_rate_init=float(best["Learning Rate"]),
    batch_size=int(best["Batch Size"]),
    max_iter=int(best["Epochs"]),
    random_state=0
)

best_model.fit(Xtr, ytr)
best_pred = best_model.predict(Xte)

# 10. Display confusion matrix
cm = confusion_matrix(yte, best_pred)

print("\nConfusion Matrix:")
print(cm)

# 11. Classification report
print("\nClassification Report:")
print(classification_report(yte, best_pred, zero_division=0))
