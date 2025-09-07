# Product Review Sentiment Analyzer

This guide walks you through creating, training, and deploying a sentiment analysis model using Scikit-learn, NLTK, and Gradio.

_`amazon_polarity`_ Dataset is used to train model. 

**Origin:** Composed by Xiang Zhang for sentiment analysis benchmarks.  
**Composition:** Includes only positive and negative reviews (3-star reviews were excluded).  
**Size:** 1.8 million training samples and 200k testing samples per class (Negative/Positive)

## Screen
![screenshot1](screenshot/Screenshot_1.png)
![screenshot2](screenshot/Screenshot_2.png)
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




