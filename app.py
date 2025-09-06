import gradio as gr
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Preprocessing function, must be the same as used for training
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# The prediction function for Gradio
def predict_sentiment(text):
    if not text:
        return "Please enter some text."
    
    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Get the model's prediction and probabilities
    prediction = model.predict(vectorized_text)[0]
    probabilities = model.predict_proba(vectorized_text)[0]
    
    # Get all possible labels and their probabilities
    labels = model.classes_
    scores = {labels[i]: probabilities[i] for i in range(len(labels))}
    
    return scores

# Create the Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=5, label="Enter text for sentiment analysis"),
    outputs=gr.Label(num_top_classes=3),
    title="Lightweight Sentiment Analyzer",
    description="Enter a sentence and get the predicted sentiment."
)

if __name__ == "__main__":
    demo.launch()

