# from flask import Flask, request, render_template, jsonify
# import joblib
# import pandas as pd
# import os

# app = Flask(__name__)

# # Load trained model and feature columns
# MODEL_PATH = r"D:\Project\Credit\src\model.pkl"
# COLUMNS_PATH = r"D:\Project\Credit\src\model_columns.pkl"

# if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
#     raise FileNotFoundError("Model or column file not found. Ensure model.pkl and model_columns.pkl exist.")

# model = joblib.load(MODEL_PATH)
# model_columns = joblib.load(COLUMNS_PATH)

# # Preprocessing function
# def preprocess_input(input_df):
#     df = pd.get_dummies(input_df)
#     for col in model_columns:
#         if col not in df.columns:
#             df[col] = 0
#     df = df[model_columns]
#     return df

# # Home page
# @app.route('/')
# def home():
#     return render_template("index.html")

# # API endpoint (JSON input)
# @app.route('/predict', methods=['POST'])
# def predict_api():
#     input_data = request.get_json(force=True)
#     df = pd.DataFrame([input_data])
#     df = preprocess_input(df)
#     prediction = model.predict(df)
#     result = "Y" if prediction[0] == 1 else "N"
#     return jsonify({"Loan_Status": result})

# # Form endpoint (HTML form)
# @app.route('/predict_form', methods=['POST'])
# def predict_form():
#     form_data = request.form.to_dict()
#     df = pd.DataFrame([form_data])
    
#     # Convert numeric fields from strings to float
#     numeric_fields = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
#     for field in numeric_fields:
#         df[field] = df[field].astype(float)
    
#     df = preprocess_input(df)
#     prediction = model.predict(df)
#     result = "Y" if prediction[0] == 1 else "N"
#     return f"<h3>Prediction: Loan_Status = {result}</h3>"

# # Run the app
# if __name__ == '__main__':
#     app.run(debug=True, port=8085)


############################################################################################################


from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model and feature columns
MODEL_PATH = r"D:\Project\Credit_Risk\src\model.pkl"
COLUMNS_PATH = r"D:\Project\Credit_Risk\src\model_columns.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
    raise FileNotFoundError("Model or column file not found. Ensure model.pkl and model_columns.pkl exist.")

model = joblib.load(MODEL_PATH)
model_columns = joblib.load(COLUMNS_PATH)

# Preprocessing function
def preprocess_input(input_df):
    df = pd.get_dummies(input_df)
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]
    return df

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# API endpoint (JSON input)
@app.route('/predict', methods=['POST'])
def predict_api():
    input_data = request.get_json(force=True)

    if input_data['Dependents'] == '3+':
        input_data['Dependents'] = 3
    input_data['Dependents'] = int(input_data['Dependents'])

    df = pd.DataFrame([input_data])
    df = preprocess_input(df)
    prediction = model.predict(df)
    result = "Y" if prediction[0] == 1 else "N"
    return jsonify({"Loan_Status": result})


# Form endpoint (HTML form)
@app.route('/predict_form', methods=['POST'])
def predict_form():
    form_data = request.form.to_dict()

    # Convert 3+ to integer
    if form_data['Dependents'] == '3+':
        form_data['Dependents'] = 3
    form_data['Dependents'] = int(form_data['Dependents'])

    # Convert numeric fields
    numeric_fields = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for field in numeric_fields:
        form_data[field] = float(form_data[field])

    df = pd.DataFrame([form_data])
    df = preprocess_input(df)

    prediction = model.predict(df)
    result = "Y" if prediction[0] == 1 else "N"
    return f"<h3>Prediction: Loan_Status = {result}</h3>"


# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8085)
