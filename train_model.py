import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")

# Load Amazon Reviews Polarity dataset (positive/negative)
dataset = load_dataset("amazon_polarity")

# Convert to pandas DataFrame (take a subset to train faster, e.g., 100k samples)
df = pd.DataFrame(dataset["train"]).sample(100000, random_state=42)

# Labels: 1 = negative, 2 = positive → remap to 0/1
df["label"] = df["label"] - 1

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

df["cleaned_text"] = df["content"].apply(preprocess_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

# Save model + vectorizer
joblib.dump(model, "sentiment_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")

print("✅ Model and vectorizer saved successfully!")
