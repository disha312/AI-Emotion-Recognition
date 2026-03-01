import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load datasets
train_df = pd.read_csv("dataset/training.csv")
val_df = pd.read_csv("dataset/validation.csv")
test_df = pd.read_csv("dataset/test.csv")

# Combine train + validation for better training
full_train_text = pd.concat([train_df["text"], val_df["text"]])
full_train_labels = pd.concat([train_df["label"], val_df["label"]])

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

X_train_vec = vectorizer.fit_transform(full_train_text)
X_test_vec = vectorizer.transform(test_df["text"])

# Train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_vec, full_train_labels)

# Evaluate on test set
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(test_df["label"], y_pred)

print("Test Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(test_df["label"], y_pred))

# Save model and vectorizer
pickle.dump(model, open("model/emotion_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("\nModel saved successfully.")