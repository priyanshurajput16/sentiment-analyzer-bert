import streamlit as st
from transformers import pipeline
from streamlit_lottie import st_lottie
import json

# Set up the sentiment analysis pipeline using a BERT model
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Function to load Lottie animation
def load_lottie(filepath):
    with open(filepath, "r") as f:
        return json.load(f)

# Set Streamlit page configuration
st.set_page_config(page_title="AI Text Sentiment Analyzer", layout="wide")

# Heading
st.markdown("<h1 style='text-align: center; color: red;'>AI Text Sentiment Analyzer</h1>", unsafe_allow_html=True)

# Show animation (optional)
try:
    lottie_animation = load_lottie("animation.json")
    st_lottie(lottie_animation, speed=1, loop=True, quality="high", height=250)
except:
    st.warning("⚠️ Animation couldn't load. Please check 'animation.json'.")

# Input box
user_input = st.text_area("✏️ Enter your sentence:", height=100)

# Analyze Button
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence!")
    else:
        result = classifier(user_input)[0]
        label = result["label"]  # E.g., '1 star' to '5 stars'
        score = result["score"]

        # Map stars to sentiment
        if label in ['1 star', '2 stars']:
            sentiment = "NEGATIVE"
            color = "red"
        elif label == '3 stars':
            sentiment = "NEUTRAL"
            color = "orange"
        else:
            sentiment = "POSITIVE"
            color = "green"

        # Display results
        st.markdown(f"<h2 style='color:{color};'>Sentiment: {sentiment}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:deepskyblue;'>Confidence: {score:.2f}</h4>", unsafe_allow_html=True)
