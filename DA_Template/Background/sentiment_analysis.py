import streamlit as st
import joblib
import numpy as np
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def sentiment_analysis_page():
    # Load SpaCy language model
    nlp = spacy.load("en_core_web_sm")

    # Preprocess text function
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        return text.strip()

    # Define the tokenizer function using SpaCy
    def spacy_tokenizer(text):
        doc = nlp(text)
        return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    # Load the TF-IDF model and fitted vectorizer
    with open('models/X_tfidf_train.sav', 'rb') as file:
        tfidf_model_data = pickle.load(file)

    # Create the TF-IDF model
    max_features = 200
    TFIDF = TfidfVectorizer(tokenizer=spacy_tokenizer, 
                            stop_words=[], 
                            ngram_range=(1, 3), 
                            max_features=max_features)
    TFIDF.fit(tfidf_model_data)  # Fit the TF-IDF model on the provided data

    # Load models and vectorizers
    def load_model_and_vectorizer():
        try:
            tfidf_vectorizer = TFIDF
            model_tfidf = joblib.load("models/myBestModel_tfidf.joblib")
            st.success("Models and vectorizers loaded successfully.")
            return tfidf_vectorizer, model_tfidf
        except Exception as e:
            st.error(f"Error loading models or vectorizers: {e}")
            return None, None

    # Predict sentiment
    def predict_sentiment(model, vectorizer, text):
        try:
            text = preprocess_text(text)
            # Transform the input text
            vectorized_text = vectorizer.transform([text])

            # Check if the input text contains recognizable words
            if vectorized_text.nnz == 0:
                st.warning("The input text contains no recognizable words from the training vocabulary.")
                return "Unknown"

            # Predict sentiment
            predicted_class = model.predict(vectorized_text)[0]

            # Return sentiment
            return "Negative" if predicted_class == 0 else "Neutral" if predicted_class == 1 else "Positive"

        except Exception as e:
            st.error(f"Error during sentiment prediction: {e}")
            return "Unknown"

    # Streamlit UI
    st.title("Sentiment Analysis Dashboard")
    st.write("Analyze the sentiment of your text using the TF-IDF model.")

    # Explanation Section
    st.subheader("What is TF-IDF?")
    st.write("""
        **TF-IDF (Term Frequency-Inverse Document Frequency)** is a statistical measure used to evaluate 
        how important a word is to a document in a collection of documents (corpus). It assigns more weight 
        to words that are unique to a document and less weight to commonly used words like "the" or "is".
        
        - **Term Frequency (TF):** Measures how often a word appears in the document.
        - **Inverse Document Frequency (IDF):** Reduces the influence of words that appear frequently across the corpus.
        
        In this tool, the TF-IDF vectorizer converts text into a numerical format, which is then fed into a pre-trained 
        machine learning model to classify sentiment as Positive, Neutral, or Negative.
    """)

    # Load models and check validity
    tfidf_vectorizer, model_tfidf = load_model_and_vectorizer()

    if all(item is not None for item in [tfidf_vectorizer, model_tfidf]):
        example_text = st.text_area("Enter the text you want to analyze:", height=150)

        if st.button("Analyze Sentiment"):
            if example_text.strip():
                sentiment = predict_sentiment(model_tfidf, tfidf_vectorizer, example_text)
                st.subheader("Analysis Result")
                st.markdown(f"**Text:** {example_text}")
                st.markdown(f"**Predicted Sentiment:** {sentiment}")
            else:
                st.warning("Please enter some text for analysis.")
    else:
        st.error("Failed to load models or vectorizers. Please check the file paths and try again.")
