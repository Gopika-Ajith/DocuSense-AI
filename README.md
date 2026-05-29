# 🧠 TextSense AI

AI-Powered News & Text Classification System

## 📌 Overview

TextSense AI is a Natural Language Processing (NLP) application that classifies news articles and text into five categories:

* 💼 Business
* 🎭 Entertainment
* 🏛 Politics
* ⚽ Sports
* 💻 Technology

The project uses TF-IDF Vectorization and Machine Learning models to accurately predict the category of a given text.

---

## 🚀 Features

* Real-time text classification
* Interactive Streamlit web application
* TF-IDF text feature extraction
* Multiple ML model comparison
* Clean and user-friendly interface

---

## 📊 Dataset

BBC News Dataset

* 2225 news articles
* 5 categories
* Business
* Entertainment
* Politics
* Sports
* Technology

---

## 🤖 Machine Learning Models

| Model                   | Accuracy |
| ----------------------- | -------- |
| Multinomial Naive Bayes | 96.40%   |
| Logistic Regression     | 97.08%   |

### Best Model

Logistic Regression (97.08%)

---

## 🛠 Technologies Used

* Python
* Pandas
* Scikit-learn
* Streamlit
* Joblib

---

## 📂 Project Structure

TextSense-AI

├── app.py

├── train.py

├── model.pkl

├── vectorizer.pkl

├── README.md

└── bbc/

---

## ▶️ Run Locally

Install dependencies:

pip install pandas scikit-learn streamlit joblib

Run:

streamlit run app.py

---

## 🎯 Result

Successfully classified news articles into five categories with a best accuracy of 97.08% using Logistic Regression and TF-IDF Vectorization.
