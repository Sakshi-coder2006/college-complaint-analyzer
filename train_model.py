
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
df = pd.read_csv("dataset/complaints.csv")

print("Dataset shape:", df.shape)
print("\nCategory counts:")
print(df["category"].value_counts())

# 2. Select input and output
X = df["complaint"]
y = df["category"]

# 3. Convert text into numbers
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(X)

# 4. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 5. Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# 6. Predict test complaints
y_pred = model.predict(X_test)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Training Successful!")
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 8. Save model and vectorizer
joblib.dump(model, "complaint_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

# 9. Save accuracy for the dashboard
with open("accuracy.txt", "w") as file:
    file.write(str(accuracy))

print("\nModel, vectorizer and accuracy saved successfully!")
