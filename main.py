import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Cache the model to avoid reloading
@st.cache_resource
def load_sentiment_model():
    return load_model('simple_rnn_imdb_optimized.h5')

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Helper functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  # Map unknown words to index 2
    valid_encoded_review = [idx if idx < 10000 else 2 for idx in encoded_review]  # Replace OOV indices with "unknown"
    padded_review = sequence.pad_sequences([valid_encoded_review], maxlen=500)
    return np.array(padded_review, dtype=np.float32)  # Ensure correct dtype

# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# Load model
model = load_sentiment_model()
st.write(f"TensorFlow Version: {tf.__version__}")  # Debug TF version

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    st.write(f"Input Shape: {preprocessed_input.shape}")  # Debug input shape

    try:
        # Make prediction
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

        # Display the result
        st.write(f'Sentiment: {sentiment}')
        st.write(f'Prediction Score: {prediction[0][0]}')

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.write('Please enter a movie review to classify.')