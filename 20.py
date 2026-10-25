# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 1. Load dataset
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target

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
X = df[["bmi", "bp", "s5"]]
y = df["target"]

# 4. Split dataset
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# 5. Apply Polynomial Regression
degrees = range(1, 7)
train_error = []
val_error = []

for degree in degrees:
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree),
        LinearRegression()
    )

    model.fit(X_train, y_train)

    train_error.append(
        mean_squared_error(y_train, model.predict(X_train))
    )
    val_error.append(
        mean_squared_error(y_val, model.predict(X_val))
    )

    print(
        f"Degree {degree} | Train MSE: {train_error[-1]:.2f} "
        f"| Validation MSE: {val_error[-1]:.2f}"
    )

# 6. Plot training and validation errors
plt.plot(degrees, train_error, "o-", label="Training Error")
plt.plot(degrees, val_error, "o-", label="Validation Error")

plt.xlabel("Polynomial Degree")
plt.ylabel("MSE (log scale)")
plt.yscale("log")
plt.title("Bias-Variance Tradeoff")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Find best polynomial degree
best_degree = list(degrees)[np.argmin(val_error)]

print("\nBest Polynomial Degree:", best_degree)
print("Lowest Validation MSE:", round(min(val_error), 2))
