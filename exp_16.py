
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")

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

# 3. Select features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Normalize data
sc = StandardScaler()
X_scaled = sc.fit_transform(X)

# 5. K-Means clustering
km = KMeans(n_clusters=5, n_init=10, random_state=0)
km_labels = km.fit_predict(X_scaled)

# 6. Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=5)
hc_labels = hc.fit_predict(X_scaled)

# 7. Evaluate models
print(
    f"KMeans       | Inertia: {km.inertia_:.2f} "
    f"| Silhouette: {silhouette_score(X_scaled, km_labels):.4f}"
)

print(
    f"Hierarchical | Silhouette: "
    f"{silhouette_score(X_scaled, hc_labels):.4f}"
)

# 8. Store results using pandas
results = pd.DataFrame({
    "Algorithm": ["K-Means", "Hierarchical"],
    "Silhouette Score": [
        silhouette_score(X_scaled, km_labels),
        silhouette_score(X_scaled, hc_labels)
    ]
})

print("\nPerformance Comparison:")
print(results.round(4))

# 9. Visualize clusters using original feature units
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].scatter(
    X.iloc[:, 0], X.iloc[:, 1],
    c=km_labels, cmap="viridis"
)
ax[0].set_title("K-Means Clustering")
ax[0].set_xlabel("Annual Income (k$)")
ax[0].set_ylabel("Spending Score")

ax[1].scatter(
    X.iloc[:, 0], X.iloc[:, 1],
    c=hc_labels, cmap="viridis"
)
ax[1].set_title("Hierarchical Clustering")
ax[1].set_xlabel("Annual Income (k$)")
ax[1].set_ylabel("Spending Score")

plt.tight_layout()
plt.show()
