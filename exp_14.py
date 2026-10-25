# Import libraries
import time
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

# 6. Compare activation functions
activations = ["logistic", "relu", "tanh"]
results = []

for a in activations:
    start = time.time()

    model = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation=a,
        max_iter=500,
        random_state=0
    )

    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    elapsed = time.time() - start

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, average="macro")
    rec = recall_score(yte, pred, average="macro")
    f1 = f1_score(yte, pred, average="macro")

    print("\nActivation:", a)
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))
    print("Epochs:", model.n_iter_)
    print("Time:", round(elapsed, 2), "seconds")

    results.append({
        "Activation": a,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Epochs": model.n_iter_,
        "Time (seconds)": elapsed
    })

# 7. Display comparison table
results_df = pd.DataFrame(results)

print("\nFinal Performance Comparison:")
print(results_df.round(4).to_string(index=False))

# 8. Find the best activation function
best_activation = results_df.loc[
    results_df["Accuracy"].idxmax(), "Activation"
]

print("\nBest Activation:", best_activation)

# 9. Train the best model
best_model = MLPClassifier(
    hidden_layer_sizes=(64,),
    activation=best_activation,
    max_iter=500,
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
print(classification_report(yte, best_pred))
