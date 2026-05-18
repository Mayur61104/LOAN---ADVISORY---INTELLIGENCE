import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import json
import openai
import os
from dotenv import load_dotenv
from nlp_parser import extract_features, get_missing_fields
from groq import Groq 
import os 
from dotenv import load_dotenv

load_dotenv("/Users/apple/Desktop/loan-advisor/apikey.env") 
client = Groq( api_key=os.getenv("GROQ_API_KEY") )

@st.cache_resource
def load_model():
    model = pickle.load(open("/Users/apple/Desktop/loan-advisor/models/model.pkl", "rb"))
    explainer = pickle.load(open("/Users/apple/Desktop/loan-advisor/models/explainer.pkl", "rb"))
    features = pickle.load(open("/Users/apple/Desktop/loan-advisor/models/features.pkl", "rb"))
    return model, explainer, features

model, explainer, feature_names = load_model()

def build_feature_vector(parsed):
    defaults = {
        "Gender": 1,
        "Married": 1,
        "Dependents": 0,
        "Education": 0,
        "Self_Employed": 0,
        "ApplicantIncome": parsed.get("applicant_income") or 5000,
        "CoapplicantIncome": parsed.get("coapplicant_income") or 0,
        "LoanAmount": parsed.get("loan_amount") or 150,
        "Loan_Amount_Term": parsed.get("loan_term_months") or 360,
        "Credit_History": parsed.get("credit_history") if parsed.get("credit_history") is not None else 1,
        "Property_Area": 2
    }

    edu_map = {"Graduate": 0, "Not Graduate": 1}
    gender_map = {"Male": 1, "Female": 0}
    married_map = {"Yes": 1, "No": 0}
    emp_map = {"Yes": 1, "No": 0}
    dep_map = {"0": 0, "1": 1, "2": 2, "3+": 3}
    area_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

    if parsed.get("education"):
        defaults["Education"] = edu_map.get(parsed["education"], 0)
    if parsed.get("gender"):
        defaults["Gender"] = gender_map.get(parsed["gender"], 1)
    if parsed.get("married"):
        defaults["Married"] = married_map.get(parsed["married"], 1)
    if parsed.get("self_employed"):
        defaults["Self_Employed"] = emp_map.get(parsed["self_employed"], 0)
    if parsed.get("dependents"):
        defaults["Dependents"] = dep_map.get(parsed["dependents"], 0)
    if parsed.get("property_area"):
        defaults["Property_Area"] = area_map.get(parsed["property_area"], 2)

    return pd.DataFrame([defaults])[feature_names]

def get_shap_explanation(feature_vector):

        shap_values = explainer.shap_values(feature_vector)

        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        else:
            sv = shap_values[0]

        pairs = sorted(
            zip(feature_names, sv),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]

        lines = []

        for name, value in pairs:

            direction = (
                "helped"
                if value > 0
                else "hurt"
            )

            lines.append(
                f"{name} {direction} your application "
                f"(impact: {value:.2f})"
            )

        return lines

def generate_response(user_msg, prediction, probability, shap_lines, parsed):
        decision = "APPROVED" if prediction == 1 else "NOT APPROVED"
        conf = f"{probability * 100:.0f}%"
        shap_text = "\n".join(shap_lines)

        prompt = f"""A small business owner asked: "{user_msg}"

    Based on ML analysis:
    - Decision: {decision}
    - Confidence: {conf}
    - Key factors:
    {shap_text}

    Write a warm, professional 3-4 sentence response:
    1. State the decision clearly
    2. Explain the top reason in simple terms
    3. If rejected, give one specific actionable tip to improve eligibility
    4. Keep it conversational, not robotic"""
        response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
        )
        return response.choices[0].message.content
st.set_page_config(page_title="Loan Advisor", page_icon="🏦")
st.title("Financial Loan Advisor")
st.caption("Describe your business and loan needs in plain English")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("e.g. My shop earns 60k/month, credit score is good, need 2 lakh loan"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your application..."):
            parsed = extract_features(prompt)
            missing = get_missing_fields(parsed)

            if missing:
                reply = f"I need a bit more info. Could you tell me your {', '.join(missing)}?"
            else:
                fv = build_feature_vector(parsed)
                pred = model.predict(fv)[0]
                prob = model.predict_proba(fv)[0][1]
                shap_lines = get_shap_explanation(fv)
                reply = generate_response(prompt, pred, prob, shap_lines, parsed)

        st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})