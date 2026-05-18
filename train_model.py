import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import shap
import pickle

df = pd.read_csv("/Users/apple/Desktop/loan-advisor/data/train.csv")

df["Gender"].fillna("Male", inplace=True)
df["Married"].fillna("Yes", inplace=True)
df["Dependents"].fillna("0", inplace=True)
df["Self_Employed"].fillna("No", inplace=True)
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(360.0, inplace=True)
df["Credit_History"].fillna(1.0, inplace=True)

le = LabelEncoder()
cat_cols = ["Gender","Married","Dependents",
            "Education","Self_Employed","Property_Area"]
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

features = ["Gender","Married","Dependents","Education",
            "Self_Employed","ApplicantIncome","CoapplicantIncome",
            "LoanAmount","Loan_Amount_Term","Credit_History",
            "Property_Area"]

X = df[features]
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, preds):.2%}")
print(classification_report(y_test, preds))

explainer = shap.TreeExplainer(model)

pickle.dump(model, open("/Users/apple/Desktop/loan-advisor/models/model.pkl", "wb"))
pickle.dump(explainer, open("/Users/apple/Desktop/loan-advisor/models/explainer.pkl", "wb"))
pickle.dump(features, open("/Users/apple/Desktop/loan-advisor/models/features.pkl", "wb"))

print("Model, explainer, and features saved.")