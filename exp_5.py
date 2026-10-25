
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

# 1. Load dataset
cats = [
    "sci.space",
    "rec.sport.hockey",
    "comp.graphics",
    "talk.politics.misc"
]

tr = fetch_20newsgroups(
    subset="train",
    categories=cats
)

te = fetch_20newsgroups(
    subset="test",
    categories=cats
)

# 2. Create DataFrames and inspect data
df_train = pd.DataFrame({
    "text": tr.data,
    "target": tr.target
})

df_test = pd.DataFrame({
    "text": te.data,
    "target": te.target
})

print("Training data:")
print(df_train.head())

print("\nDataset shape:", df_train.shape)
print("\nMissing values:\n", df_train.isnull().sum())

# 3. Clean data
df_train = df_train.dropna()
df_test = df_test.dropna()

df_train = df_train.drop_duplicates(subset=["text"])

print("\nTraining shape after cleaning:", df_train.shape)

# 4. Convert text into word-count features
cv = CountVectorizer(stop_words="english")

Xtr = cv.fit_transform(df_train["text"])
Xte = cv.transform(df_test["text"])

ytr = df_train["target"].to_numpy()
yte = df_test["target"].to_numpy()

print("\nTraining matrix:", Xtr.shape)
print("Testing matrix:", Xte.shape)

# 5. Train models for different alpha values
alphas = [1e-10, 0.01, 0.1, 1, 10]

results = []

for a in alphas:
    m = MultinomialNB(alpha=a)
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
        f"alpha={a:<8} | Accuracy: {acc:.4f} "
        f"| Precision: {prec:.4f} "
        f"| Recall: {rec:.4f} "
        f"| F1: {f1:.4f}"
    )

    results.append({
        "Alpha": a,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

# 6. Display results using pandas
results_df = pd.DataFrame(results)

print("\nFinal comparison:")
print(results_df.round(4))

# 7. Plot accuracy for different alpha values
plt.figure(figsize=(8, 5))

plt.plot(
    results_df["Alpha"],
    results_df["Accuracy"],
    marker="o"
)

plt.xscale("log")
plt.xlabel("Alpha (log scale)")
plt.ylabel("Accuracy")
plt.title("Effect of Alpha on Multinomial Naive Bayes")
plt.grid(True)
plt.show()

# 8. Confusion matrix for the best alpha
best_alpha = results_df.loc[
    results_df["Accuracy"].idxmax(), "Alpha"
]

best_model = MultinomialNB(alpha=best_alpha)
best_model.fit(Xtr, ytr)
best_pred = best_model.predict(Xte)

cm = confusion_matrix(yte, best_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=tr.target_names,
    yticklabels=tr.target_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (alpha={best_alpha})")
plt.tight_layout()
plt.show()

print("\nClassification report:")
print(classification_report(
    yte,
    best_pred,
    target_names=tr.target_names,
    zero_division=0
))
