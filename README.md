# 🏦 Financial Loan Advisor Intelligence System

An end-to-end AI-powered financial advisory platform that combines Large Language Models (LLMs), Machine Learning, Explainable AI, and conversational interfaces to simulate an intelligent loan advisor.

The system allows users to describe their financial situation in natural language. An LLM extracts structured loan application features, an XGBoost model predicts loan eligibility, SHAP explains the prediction, and another LLM generates personalized financial recommendations.

The application is deployed using Streamlit and containerized with Docker for portable deployment.

---

# 🚀 Features

* Conversational loan advisory system
* Natural language financial input
* LLM-based feature extraction
* XGBoost loan eligibility prediction
* SHAP explainability integration
* Human-readable financial recommendations
* Streamlit interactive interface
* Dockerized deployment
* Docker Hub image distribution
* End-to-end AI + ML pipeline

---

# 🧠 System Architecture

```text
User Input (Natural Language)
                │
                ▼
      Groq LLM Feature Extraction
                │
                ▼
      Structured Financial Features
                │
                ▼
      XGBoost Loan Prediction Model
                │
                ▼
         SHAP Explainability
                │
                ▼
      LLM Financial Recommendation
                │
                ▼
         Streamlit Frontend
```

---

# 📊 Model Performance

| Metric         | Score              |
| -------------- | ------------------ |
| Recall         | 93%                |
| Model          | XGBoost Classifier |
| Explainability | SHAP               |

The model was trained using structured financial datasets and optimized through feature engineering and hyperparameter tuning.

---

# 📌 Tech Stack

### AI & LLMs

* Groq API
* Prompt Engineering
* NLP Parsing
* Conversational AI

### Machine Learning

* XGBoost
* Scikit-learn
* SHAP

### Backend & Deployment

* Python
* Streamlit
* Docker
* Docker Hub

### Data Processing

* Pandas
* NumPy

---

# 🔍 NLP Feature Extraction

The system extracts structured loan application information from conversational user inputs.

Extracted features include:

* Applicant Income
* Co-applicant Income
* Loan Amount
* Loan Term
* Credit History
* Education
* Employment Status
* Property Area
* Marital Status
* Dependents

Example:

> "My shop earns around 60k per month, I need a 2 lakh loan to expand inventory, my credit history is good and I live in an urban area."

Automatically becomes structured model-ready features for inference.

---

# 📊 Explainable AI

SHAP (SHapley Additive Explanations) is used to explain model predictions.

The system identifies:

* Positive contributing factors
* Negative contributing factors
* Relative feature importance
* Financial indicators influencing approval decisions

This improves transparency and trustworthiness of recommendations.

---

# 🖥️ Running Locally

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

Create:

```text
apikey.env
```

Add:

```text
GROQ_API_KEY=your_api_key_here
```

## Run Application

```bash
streamlit run app.py
```

---

# 🐳 Docker Deployment

## Pull Docker Image

```bash
docker pull mayur61104/loan-advisor:latest
```

## Run Container

```bash
docker run --env-file apikey.env -p 8501:8501 mayur61104/loan-advisor:latest
```

Access the application at:

```text
http://localhost:8501
```

---

## Build Docker Image Locally

```bash
docker build -t loan-advisor .
```

Run:

```bash
docker run --env-file apikey.env -p 8501:8501 loan-advisor
```

## Docker Hub

Docker image:

docker pull mayuricreates/loan-advisor-bot:latest

Docker Hub Repository:
https://hub.docker.com/r/mayuricreates/loan-advisor-bot

---

# 🔐 Security Notice

Do not commit API keys to GitHub.

Store credentials locally using:

```text
apikey.env
```

and provide them to the application through environment variables.

---

# 📁 Project Structure

```text
loan-advisor/
│
├── app.py
├── nlp_parser.py
├── train_model.py
├── Dockerfile
├── requirements.txt
├── README.md
├── apikey.env.example
│
├── models/
│   ├── model.pkl
│   ├── explainer.pkl
│   └── features.pkl
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
```

---

# 🔮 Future Improvements

* Agentic financial workflows
* RAG-based financial knowledge retrieval
* PDF financial statement analysis
* Multilingual support
* Voice-enabled financial advisor
* Cloud-native deployment
* User authentication
* Vector database integration
* Personalized financial planning

---

# 📬 Author

Developed as an applied AI/ML project focused on combining conversational AI, explainable machine learning, and financial intelligence into a production-oriented advisory platform.
