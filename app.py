import gradio as gr
import joblib
import re
import nltk
from nltk.corpus import stopwords

# Load trained model + vectorizer
model = joblib.load("sentiment_model.joblib")
vectorizer = joblib.load("tfidf_vectorizer.joblib")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    stop_words = set(stopwords.words("english"))
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# Prediction function
def predict_sentiment(text, threshold=10):
    if not text or not text.strip():
        return "Please enter some text.", {}

    cleaned_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([cleaned_text])

    probabilities = model.predict_proba(vectorized_text)[0]

    labels = ["Negative", "Positive"]
    confidence_scores = {
        labels[i]: round(probabilities[i] * 100, 2) for i in range(len(labels))
    }

    # Find best and second best
    sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)
    best_label, best_score = sorted_scores[0]
    second_label, second_score = sorted_scores[1]

    # If the gap is small, call it Neutral
    if abs(best_score - second_score) < threshold:
        predicted_label = "Neutral"
    else:
        predicted_label = best_label

    return predicted_label, confidence_scores

# Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(label="Review",lines=3, placeholder="Write your product review here..."),
    outputs=[
        gr.Label(num_top_classes=2, label="Predicted Sentiment"),
        gr.JSON(label="Confidence Scores")
    ],
    title="Product Review Sentiment Analyzer",
    description="Enter a product review and see whether it is positive or negative, with confidence scores."
)

if __name__ == "__main__":
    demo.launch()
