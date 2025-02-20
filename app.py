import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords

def get_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)
    
    # Remove stopwords from the text
    filtered_words = [word for word in words if word.lower() not in stop_words]
    
    # Reconstruct the text without stopwords
    filtered_text = " ".join(filtered_words)
    return filtered_text


@st.cache_resource
def load_model():
    with open("sentiment_pipeline.pkl", "rb") as f:
        return pickle.load(f)

def predict(text: str, model = load_model()):
    processed_text = get_stopwords(text)
    sentiment = model.predict([processed_text])
    return sentiment[0]

# Streamlit UI
st.title("Customer Sentiment Analysis")
st.write("Enter text below to analyze sentiment:")

user_input = st.text_area("Input Text:", "Type your review here...")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        prediction = predict(user_input)
        st.write("### Sentiment Prediction:")
        st.write(f"**{prediction}**")
    else:
        st.warning("Please enter some text before analyzing.")