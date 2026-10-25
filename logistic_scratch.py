import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def train(X, y, lr=0.1, epochs=1000, lam=0.1):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        error = p - y
        dw = (1 / n) * (error @ X) + (lam / n) * w
        db = (1 / n) * np.sum(error)
        w = w - lr * dw
        b = b - lr * db
    return w, b

def predict(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)

def evaluate(name, y_true, y_pred):
    print(name,
          "Accuracy:", round(accuracy_score(y_true, y_pred), 4),
          "Precision:", round(precision_score(y_true, y_pred), 4),
          "Recall:", round(recall_score(y_true, y_pred), 4),
          "F1:", round(f1_score(y_true, y_pred), 4))

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42)

# WITHOUT scaling
w, b = train(X_train, y_train)
evaluate("Unscaled", y_test, predict(X_test, w, b))

# WITH scaling
sc = StandardScaler().fit(X_train)
Xtr = sc.transform(X_train)
Xte = sc.transform(X_test)
w2, b2 = train(Xtr, y_train)
evaluate("Scaled  ", y_test, predict(Xte, w2, b2))
