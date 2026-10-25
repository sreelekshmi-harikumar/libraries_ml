# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 1. Load dataset
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
print("\nDataset shape:", df.shape)
df.info()
print("\nMissing values:")
print(df.isnull().sum())

# 2. Clean dataset
df = df.dropna()
df = df.drop_duplicates()

# 3. Select features for clustering
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# 4. Normalize data
X_scaled = StandardScaler().fit_transform(X)

# 5. K-Means clustering
km = KMeans(n_clusters=5, n_init=10, random_state=0)
km_labels = km.fit_predict(X_scaled)

# 6. Hierarchical clustering
hc = AgglomerativeClustering(n_clusters=5)
hc_labels = hc.fit_predict(X_scaled)

# 7. Inertia (added): sum of squared distances to each cluster's centre
def inertia(X, labels):
    return sum(((X[labels == k] - X[labels == k].mean(axis=0)) ** 2).sum()
               for k in np.unique(labels))

# 8. Evaluate and compare
results = pd.DataFrame({
    "Algorithm": ["K-Means", "Hierarchical"],
    "Inertia": [km.inertia_, inertia(X_scaled, hc_labels)],
    "Silhouette": [silhouette_score(X_scaled, km_labels),
                   silhouette_score(X_scaled, hc_labels)]
})
print("\nPerformance Comparison:")
print(results.round(4))

# 9. Visualization (simplified)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=km_labels, cmap="viridis")
plt.title("K-Means Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")

plt.subplot(1, 2, 2)
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=hc_labels, cmap="viridis")
plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")

plt.tight_layout()
plt.show()
