📦 milestone2
│
├── agri_yield_training.py           # Model training and evaluation
├── insightful_eda_feature_selection.py  # Exploratory Data Analysis & Feature Selection
│
├── results.png                      # Comparison visualization of models (if created)
├── feature_importance.png           # Importance of features from Random Forest/XGBoost
├── predicted_vs_actual.png          # Actual vs Predicted yield visualization
│
├── metrics.txt                      # Model performance metrics
├── README.md                        # Project documentation (this file)
1️⃣ Data Understanding & Preprocessing

Dataset: processed_crop_data.csv

Columns include soil nutrients (N, P, K), environmental parameters (temperature, humidity, rainfall), and pH values.

Missing values and data types were checked and cleaned.

2️⃣ Exploratory Data Analysis (EDA)

File: insightful_eda_feature_selection.py

Generated correlation matrix to identify relationships between variables.

Visualized feature-target relationships using scatter plots.

Applied SelectKBest (f_regression) to score feature importance statistically.

3️⃣ Model Training & Evaluation

File: agri_yield_training.py

Splits data into training and test sets (80–20 ratio).

Scales data using StandardScaler.

Trains and compares three regression models:

🌲 Random Forest Regressor

🚀 XGBoost Regressor

📈 Linear Regression

Metrics used:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score