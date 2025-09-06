# 📝 Lightweight Sentiment Analysis 

This guide walks you through creating, training, and deploying a lightweight sentiment analysis model using Scikit-learn, NLTK, and Gradio.

## Project Setup and Run:
1. Install Dependencies:  
   ```sh
    pip install -r requirements.txt
   ```

2. Download stop word if nltk_data folder is empty.  
   ```sh
   python download_nltk_stop_words.py
   ```
   
3. Train the Sentiment Model This creates two files:sentiment_model.joblib and tfidf_vectorizer.joblib  
   ```sh
   python train_model.py
   ```

4. Run the App  
   ```sh
   python app.py
   ```




