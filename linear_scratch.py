import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load data, pick BMI
data = load_diabetes()
X = data.data[:, 2].reshape(-1, 1)
y = data.target

# 2. Split FIRST
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 3. Scale: learn from train, apply to both, flatten
sc = StandardScaler().fit(X_train)
X_train = sc.transform(X_train).ravel()
X_test = sc.transform(X_test).ravel()
n = len(y_train)

# 4. Initialise
w, b = 0.0, 0.0
lr0, decay, epochs = 0.1, 0.001, 1000
cost_history = []

# 5. Gradient descent on the TRAINING data only
for epoch in range(epochs):
    y_pred = w * X_train + b
    error = y_pred - y_train

    dw = (2 / n) * np.sum(error * X_train)
    db = (2 / n) * np.sum(error)

    lr = lr0 / (1 + decay * epoch)
    w = w - lr * dw
    b = b - lr * db

    cost_history.append(np.mean(error ** 2))

# 6. Evaluate on both sets
train_mse = np.mean((w * X_train + b - y_train) ** 2)
test_mse = np.mean((w * X_test + b - y_test) ** 2)
print(f"Learned w = {w:.4f}")
print(f"Learned b = {b:.4f}")
print(f"Train MSE = {train_mse:.4f}")
print(f"Test MSE  = {test_mse:.4f}")

# 7. Plot
plt.plot(cost_history)
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")
plt.title("Cost vs Epochs")
plt.grid(True)
plt.show()
