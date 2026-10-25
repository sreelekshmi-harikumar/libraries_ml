# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load dataset (manual DataFrame construction)
diabetes = load_diabetes()

df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df["target"] = diabetes.target

# 3. Handle missing values and duplicates
print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna()
df = df.drop_duplicates()

print("\nShape after cleaning:", df.shape)

# 4. Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

# 5. Train-test split
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# 6. Standardize features
scaler = StandardScaler()

Xtr = scaler.fit_transform(Xtr)
Xte = scaler.transform(Xte)

# 7. Define models
models = {
    "Linear": LinearRegression(),
    "Ridge": RidgeCV(alphas=[0.01, 0.1, 1, 10], cv=5),
    "Lasso": LassoCV(alphas=[0.01, 0.1, 1, 10], cv=5, max_iter=10000)
}

# 8. Train, predict and evaluate
results = []

for name, model in models.items():
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)

    mse = mean_squared_error(yte, pred)
    r2 = r2_score(yte, pred)

    print(f"{name:<7} | MSE: {mse:.2f} | R2: {r2:.4f}")

    if name == "Ridge" or name == "Lasso":
        print("Selected alpha:", model.alpha_)

    results.append({
        "Model": name,
        "MSE": mse,
        "R2 Score": r2
    })

# 9. Results in a pandas DataFrame
results_df = pd.DataFrame(results)

print("\nFinal Comparison:")
print(results_df.round(4))
