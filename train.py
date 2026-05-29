import os
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
dataset_path = "bbc"

texts = []
labels = []

for category in os.listdir(dataset_path):

    category_path = os.path.join(dataset_path, category)

    if not os.path.isdir(category_path):
        continue

    for file in os.listdir(category_path):

        file_path = os.path.join(category_path, file)

        with open(file_path, "r", encoding="latin-1") as f:
            text = f.read()

        texts.append(text)
        labels.append(category)

# Create DataFrame
df = pd.DataFrame({
    "text": texts,
    "category": labels
})

print(f"Total Documents: {len(df)}")

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english")

X = vectorizer.fit_transform(df["text"])
y = df["category"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n" + "="*50)
print("MULTINOMIAL NAIVE BAYES")
print("="*50)

# Train Naive Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict
nb_predictions = nb_model.predict(X_test)

# Accuracy
nb_accuracy = accuracy_score(y_test, nb_predictions)

print(f"Accuracy: {nb_accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, nb_predictions))

print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

# Predict
lr_predictions = lr_model.predict(X_test)

# Accuracy
lr_accuracy = accuracy_score(y_test, lr_predictions)

print(f"Accuracy: {lr_accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, lr_predictions))

print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

print(f"Multinomial Naive Bayes : {nb_accuracy * 100:.2f}%")
print(f"Logistic Regression     : {lr_accuracy * 100:.2f}%")

# Save best model
if lr_accuracy > nb_accuracy:
    best_model = lr_model
    best_name = "Logistic Regression"
else:
    best_model = nb_model
    best_name = "Multinomial Naive Bayes"

joblib.dump(best_model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(f"\nBest Model: {best_name}")
print("Model saved successfully!")