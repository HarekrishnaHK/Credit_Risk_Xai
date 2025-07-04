# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import mlflow
# import mlflow.sklearn
# import joblib

# # Load preprocessed data
# df = pd.read_csv(r"D:\Project\Credit_Risk\data\processed\cleaned_data.csv")

# # Split features and target
# X = df.drop('Loan_Status', axis=1)
# y = df['Loan_Status']

# # Save feature columns
# joblib.dump(list(X.columns), r"D:\Project\Credit_Risk\src\model_columns.pkl")

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define model and parameters
# model = DecisionTreeClassifier(max_depth=5, random_state=42)
# params = {"max_depth": 5, "random_state": 42}

# # MLflow experiment
# mlflow.set_experiment("credit-risk-xai")

# with mlflow.start_run():
#     mlflow.log_params(params)

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     mlflow.log_metric("accuracy", accuracy)

#     mlflow.sklearn.log_model(
#         sk_model=model,
#         name="decision_tree_model",
#         input_example=X_train.iloc[:5],
#         registered_model_name="credit-risk-decision-tree"
#     )

#     print(f"✅ Model logged to MLflow with accuracy: {accuracy:.4f}")

# # Save model locally
# joblib.dump(model, r"D:\Project\Credit_Risk\src\model.pkl")


#########################################################################################################################



# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score
# import mlflow
# import mlflow.sklearn
# import joblib
# import os

# # Load preprocessed dataset
# df = pd.read_csv(r"D:\Project\Credit_Risk\data\processed\cleaned_data.csv")

# # Split features and target
# X = df.drop('Loan_Status', axis=1)
# y = df['Loan_Status']

# # Save feature columns for prediction pipeline
# os.makedirs(r"D:\Project\Credit_Risk\src", exist_ok=True)
# joblib.dump(list(X.columns), r"D:\Project\Credit_Risk\src\model_columns.pkl")

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Define model and hyperparameters
# model = DecisionTreeClassifier(max_depth=5, random_state=42)
# params = {"max_depth": 5, "random_state": 42}

# # Set MLflow experiment
# mlflow.set_experiment("credit-risk-xai")

# # Train and log with MLflow
# with mlflow.start_run():
#     mlflow.log_params(params)
    
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
    
#     accuracy = accuracy_score(y_test, y_pred)
#     mlflow.log_metric("accuracy", accuracy)
    
#     mlflow.sklearn.log_model(
#         sk_model=model,
#         name="decision_tree_model",
#         input_example=X_train.iloc[:5],
#         registered_model_name="credit-risk-decision-tree"
#     )

#     print(f"✅ Model logged to MLflow with accuracy: {accuracy:.4f}")

# # Save model locally
# joblib.dump(model, r"D:\Project\Credit_Risk\src\model.pkl")


#################################################################################################################################


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import joblib
import os

# Load data
df = pd.read_csv(r"D:\Project\Credit_Risk\data\processed\cleaned_data.csv")
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Save feature columns
os.makedirs(r"D:\Project\Credit_Risk\src", exist_ok=True)
joblib.dump(list(X.columns), r"D:\Project\Credit_Risk\src\model_columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Grid Search
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy'
)

grid_search.fit(X_train, y_train)

# Best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict and evaluate
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# MLflow logging
mlflow.set_experiment("credit-risk-xai")

with mlflow.start_run():
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(
        sk_model=best_model,
        name="decision_tree_model",
        input_example=X_train.iloc[:5],
        registered_model_name="credit-risk-decision-tree"
    )

    print(f"✅ Best Parameters: {best_params}")
    print(f"✅ Model logged with accuracy: {accuracy:.4f}")

# Save best model
joblib.dump(best_model, r"D:\Project\Credit_Risk\src\model.pkl")
