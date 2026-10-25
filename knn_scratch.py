import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

# 1. Load the dataset
digits = load_digits()
X = digits.data
y = digits.target

# 2. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Euclidean distance
def euclidean(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# 4. Manhattan distance
def manhattan(a, b):
    return np.sum(np.abs(a - b))

# 5. Predict the digit using KNN
def knn_predict(x, X_train, y_train, k, metric):
    distances = []

    for i in range(len(X_train)):
        d = metric(x, X_train[i])
        distances.append((d, y_train[i]))

    distances.sort(key=lambda item: item[0])
    neighbors = distances[:k]
    labels = [item[1] for item in neighbors]
    prediction = max(set(labels), key=labels.count)

    return prediction

# 6. Test different K values and both metrics
k_values = [1, 3, 5, 7, 9, 15, 25]
euclid_acc = []
manhattan_acc = []

for metric_name, metric in [("Euclidean", euclidean), ("Manhattan", manhattan)]:
    for k in k_values:

        start = time.time()
        predictions = []

        for x in X_test:
            pred = knn_predict(x, X_train, y_train, k, metric)
            predictions.append(pred)

        accuracy = np.mean(np.array(predictions) == y_test) * 100
        elapsed = time.time() - start

        print(
            metric_name,
            "K =", k,
            "Accuracy =", round(accuracy, 2), "%",
            "Time =", round(elapsed, 2), "seconds"
        )

        if metric_name == "Euclidean":
            euclid_acc.append(accuracy)
        else:
            manhattan_acc.append(accuracy)

# 7. Plot accuracy vs K
plt.plot(k_values, euclid_acc, marker="o", label="Euclidean")
plt.plot(k_values, manhattan_acc, marker="o", label="Manhattan")
plt.xlabel("K")
plt.ylabel("Accuracy (%)")
plt.title("KNN accuracy vs K")
plt.legend()
plt.grid(True)
plt.show()
