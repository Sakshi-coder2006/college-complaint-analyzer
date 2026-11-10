import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Read dataset
data = pd.read_csv("dataset/complaints.csv")

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

data["complaint"] = data["complaint"].apply(clean_text)

# Convert text into numerical values
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["complaint"])

# Display results
print("TF-IDF Matrix:")
print(X.toarray())

print("\nMatrix Shape:")
print(X.shape)

print("\nWords:")
print(vectorizer.get_feature_names_out())