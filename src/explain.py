import shap
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os

# Paths
MODEL_PATH = r"D:\Project\Credit_Risk\src\model.pkl"
COLUMNS_PATH = r"D:\Project\Credit_Risk\src\model_columns.pkl"
DATA_PATH = r"D:\Project\Credit_Risk\data\processed\cleaned_data.csv"

# Load model and feature columns
model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)

# Load preprocessed data
df = pd.read_csv(DATA_PATH)
X = df.drop("Loan_Status", axis=1)

# Ensure column order and convert to float
X = X[columns].astype(float)

# SHAP explanation
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Save SHAP summary plot
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.savefig("shap_summary1.png", bbox_inches='tight')
print("✅ SHAP summary plot saved as shap_summary1.png")
