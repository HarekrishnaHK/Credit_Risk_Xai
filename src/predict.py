import pandas as pd
import joblib
import argparse
import sys
import os

# Set model paths
MODEL_PATH = r"D:\Project\Credit\src\model.pkl"
COLUMNS_PATH = r"D:\Project\Credit\src\model_columns.pkl"

def preprocess_input(input_dict):
    # Handle special cases
    if input_dict['Dependents'] == '3+':
        input_dict['Dependents'] = 3
    input_dict['Dependents'] = int(input_dict['Dependents'])

    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)

    # Load training columns
    train_cols = joblib.load(COLUMNS_PATH)
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[train_cols].astype(float)
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Gender', default='Male')
    parser.add_argument('--Married', default='Yes')
    parser.add_argument('--Dependents', default='0')
    parser.add_argument('--Education', default='Not Graduate')
    parser.add_argument('--Self_Employed', default='No')
    parser.add_argument('--ApplicantIncome', type=float, default=2333)
    parser.add_argument('--CoapplicantIncome', type=float, default=1516)
    parser.add_argument('--LoanAmount', type=float, default=95)
    parser.add_argument('--Loan_Amount_Term', type=float, default=360)
    parser.add_argument('--Credit_History', type=float, default=1.0)
    parser.add_argument('--Property_Area', default='Urban')

    args = parser.parse_args()
    input_data = vars(args)

    # Load model
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found.")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)

    # Preprocess
    df = preprocess_input(input_data)

    # Predict
    prediction = model.predict(df)
    result = "Y" if prediction[0] == 1 else "N"
    print(f"✅ Prediction: Loan_Status = {result}")

if __name__ == '__main__':
    main()
