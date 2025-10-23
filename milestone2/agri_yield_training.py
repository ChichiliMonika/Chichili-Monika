# insightful_eda_feature_selection.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression

# 1️⃣ Load dataset
df = pd.read_csv("processed_crop_data.csv")

# 2️⃣ Quick overview
print("✅ Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)
print("\nDescriptive statistics:\n", df.describe())

# 3️⃣ Correlation analysis (logical insight for feature selection)
plt.figure(figsize=(10, 6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Insight:
# Highly correlated features with the target (label_encoded or yield) can be prioritized.
# Features with near-zero correlation may be considered less important.

# 4️⃣ Visualizing relationships with target
target_col = "label_encoded"  # change to your target
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

for feature in features:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=df[feature], y=df[target_col])
    plt.title(f"{feature} vs {target_col}")
    plt.show()

# 5️⃣ Feature selection using statistical method (SelectKBest)
X = df[features]
y = df[target_col]

# Using f_regression for numerical features
selector = SelectKBest(score_func=f_regression, k='all')  # score all features
selector.fit(X, y)

# Display scores
feature_scores = pd.DataFrame({'Feature': features, 'Score': selector.scores_})
feature_scores = feature_scores.sort_values(by='Score', ascending=False)
print("\n✅ Feature Importance based on statistical test:")
print(feature_scores)

# Logical selection explanation:
# - Features with higher scores contribute more to the target.
# - These can be prioritized for model training.
# - Less important features can be dropped to reduce noise and improve model performance.
# agri_yield_training.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pickle

# Load your processed dataset
df = pd.read_csv("processed_crop_data.csv")

# Separate features and target
# Make sure 'yield' is your target column name — change if it’s different
# Separate features and target
X = df.drop("label_encoded", axis=1)
y = df["label_encoded"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    max_depth=12,
    min_samples_split=4
)
model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred = model.predict(X_test_scaled)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("✅ Model Evaluation Results:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the model and scaler for later use
with open("yield_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("🎉 Model and scaler saved successfully!")
from sklearn.model_selection import train_test_split

# Example if not already done
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load your processed dataset
df = pd.read_csv("processed_crop_data.csv")

# Separate features and target
X = df.drop("label_encoded", axis=1)
y = df["label_encoded"]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=4, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
    "Linear Regression": LinearRegression()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"✅ {name} Evaluation:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.2f}\n")

# Compare models and select the best one (highest R² or lowest RMSE)
best_model_name = min(results, key=lambda x: results[x]["RMSE"])  # or use max R²
best_model = models[best_model_name]
print(f"🎯 Best Model Selected: {best_model_name}")

# Save the best model and scaler
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("🎉 Best model and scaler saved successfully!")

import matplotlib.pyplot as plt

# Feature importance from Random Forest or XGBoost
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_names = X.columns
    
    plt.figure(figsize=(8,5))
    plt.barh(feature_names, importances)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.savefig("feature_importance.png")
    plt.show()
# Predicted vs Actual Plot
plt.figure(figsize=(6,6))
plt.scatter(y_test, best_model.predict(X_test_scaled))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Yield")
plt.ylabel("Predicted Yield")
plt.title("Predicted vs Actual Yield")
plt.savefig("predicted_vs_actual.png")
plt.show()
with open("metrics.txt", "w") as f:
    for model_name, res in results.items():
        f.write(f"{model_name}:\n")
        f.write(f"MAE: {res['MAE']:.2f}, RMSE: {res['RMSE']:.2f}, R2: {res['R2']:.2f}\n\n")