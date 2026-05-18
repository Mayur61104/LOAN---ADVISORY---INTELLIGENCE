from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv("/Users/apple/Desktop/loan-advisor/apikey.env")

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)


SYSTEM_PROMPT = """
You extract loan application data from user messages.
Return ONLY valid JSON. No extra text. No markdown.

Return this exact structure:
{
  "applicant_income": number or null,
  "coapplicant_income": number or null,
  "loan_amount": number or null,
  "loan_term_months": number or null,
  "credit_history": 1 or 0 or null,
  "education": "Graduate" or "Not Graduate" or null,
  "self_employed": "Yes" or "No" or null,
  "property_area": "Urban" or "Semiurban" or "Rural" or null,
  "gender": "Male" or "Female" or null,
  "married": "Yes" or "No" or null,
  "dependents": "0" or "1" or "2" or "3+" or null
}

For credit_history:
1 means good credit,
0 means bad credit.

If user says amounts in lakhs,
convert to thousands (1 lakh = 100).

If any field is not mentioned,
return null for it.
"""

import json

def extract_features(user_message):

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0
    )

    raw = response.choices[0].message.content

    print("RAW OUTPUT:")
    print(repr(raw))

    if raw is None or raw.strip() == "":
        return {}

    raw = raw.replace("```json", "")
    raw = raw.replace("```", "")
    raw = raw.strip()

    try:
        return json.loads(raw)

    except Exception as e:

        print("JSON ERROR:")
        print(e)

        print("BAD JSON:")
        print(raw)

        return {}
def get_missing_fields(features_dict):

    important = [
        "applicant_income",
        "loan_amount",
        "credit_history"
    ]

    return [
        f for f in important
        if features_dict.get(f) is None
    ]