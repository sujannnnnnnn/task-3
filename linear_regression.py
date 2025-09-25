# linear_regression_housing.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("Housing.csv")

print("Dataset Shape:", df.shape)
print(df.head(), "\n")

# Check for missing values
print("Missing values:\n", df.isnull().sum(), "\n")

# ===============================
# 2. SIMPLE LINEAR REGRESSION
# Example: area (sqft) -> price
# ===============================
X = df[["area"]]   # independent variable
y = df["price"]    # dependent variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

lr_simple = LinearRegression()
lr_simple.fit(X_train, y_train)

y_pred_simple = lr_simple.predict(X_test)

print("---- Simple Linear Regression ----")
print("Intercept:", lr_simple.intercept_)
print("Coefficient:", lr_simple.coef_[0])
print("MAE:", mean_absolute_error(y_test, y_pred_simple))
print("MSE:", mean_squared_error(y_test, y_pred_simple))
print("R²:", r2_score(y_test, y_pred_simple), "\n")

# Plot regression line
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred_simple, color="red", linewidth=2, label="Predicted")
plt.xlabel("Area (sqft)")
plt.ylabel("Price")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()

# ===============================
# 3. MULTIPLE LINEAR REGRESSION
# Example: use area, bedrooms, bathrooms to predict price
# ===============================
features = ["area", "bedrooms", "bathrooms"]  # adjust if dataset has other cols
X_multi = df[features]
y_multi = df["price"]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

lr_multi = LinearRegression()
lr_multi.fit(X_train_m, y_train_m)

y_pred_multi = lr_multi.predict(X_test_m)

print("---- Multiple Linear Regression ----")
print("Intercept:", lr_multi.intercept_)
print("Coefficients:", dict(zip(features, lr_multi.coef_)))
print("MAE:", mean_absolute_error(y_test_m, y_pred_multi))
print("MSE:", mean_squared_error(y_test_m, y_pred_multi))
print("R²:", r2_score(y_test_m, y_pred_multi), "\n")

# Plot Actual vs Predicted
plt.scatter(y_test_m, y_pred_multi, color="purple")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Multiple Linear Regression")
plt.show()
