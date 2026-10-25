
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

# 3. Separate features
X = df.drop("target", axis=1)

# 4. Normalize features
sc = StandardScaler()
X = sc.fit_transform(X)

# 5. Apply K-Means for different K values
ks = range(2, 16)
inertia, sil = [], []

for k in ks:
    m = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)

    inertia.append(m.inertia_)
    sil.append(silhouette_score(X, m.labels_))

    print(
        f"K={k:<3} | Inertia: {m.inertia_:.2f} "
        f"| Silhouette: {sil[-1]:.4f}"
    )

# 6. Store results in pandas DataFrame
results_df = pd.DataFrame({
    "K": list(ks),
    "Inertia": inertia,
    "Silhouette Score": sil
})

print("\nFinal Results:")
print(results_df.round(4).to_string(index=False))

# 7. Plot Inertia and Silhouette Score
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(ks, inertia, "o-")
ax[0].set_title("Elbow Method - Inertia")
ax[0].set_xlabel("Number of Clusters (K)")
ax[0].set_ylabel("Inertia")
ax[0].grid(True)

ax[1].plot(ks, sil, "o-", color="green")
ax[1].set_title("Silhouette Score")
ax[1].set_xlabel("Number of Clusters (K)")
ax[1].set_ylabel("Silhouette Score")
ax[1].grid(True)

plt.tight_layout()
plt.show()

# 8. Find K with the highest silhouette score
best_k = results_df.loc[
    results_df["Silhouette Score"].idxmax(), "K"
]

print("\nBest K based on Silhouette Score:", best_k)
