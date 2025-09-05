import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import nltk
from nltk.corpus import stopwords
import re

# Simple dataset of example sentences for training
data = {
    'text': [
        "This movie was fantastic, I loved it!",
        "The food was okay, nothing special.",
        "I hated the slow plot, it was terrible.",
        "A truly magnificent and breathtaking experience.",
        "Not my favorite, a bit boring to be honest.",
        "I would definitely watch this again.",
        "The service was incredibly slow.",
        "Perfect in every single way."
    ],
    'sentiment': ['positive', 'neutral', 'negative', 'positive', 'negative', 'positive', 'negative', 'positive']
}
df = pd.DataFrame(data)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Train a lightweight model
model = LogisticRegression(random_state=42)
model.fit(X, y)

# Save the trained model and vectorizer
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("Model and vectorizer saved successfully.")

