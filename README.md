📝 Lightweight Sentiment Analysis 

This guide walks you through creating, training, and deploying a lightweight sentiment analysis model using Scikit-learn, NLTK, and Gradio.

Project Setup:
1. Install Dependencies:
   pip install -r requirements.txt
   
2. Download stop word if nltk_data folder is empty.
   python download_nltk_stop_words.py

3.Train the Sentiment Model This creates two files:sentiment_model.joblib and tfidf_vectorizer.joblib
  python train_model.py

4. Run the App
   python app.py



