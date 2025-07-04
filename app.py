import streamlit as st
from transformers import pipeline
import json
from streamlit_lottie import st_lottie

# Streamlit page config
st.set_page_config(page_title="AI Sentiment Analyzer", layout="wide")
st.markdown("<h1 style='text-align: center; color: red;'> AI Text Sentiment Analyzer</h1>", unsafe_allow_html=True)

# Load Lottie animation
def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

lottie_animation = load_lottie("animation.json")
st_lottie(lottie_animation, speed=1, reverse=False, loop=True, quality="high", height=250)

# Load sentiment pipeline (no model.to(device)!)
classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Input box
user_input = st.text_area("✏️ Enter your sentence:", height=100)

# Button click
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence!")
    else:
        result = classifier(user_input)[0]
        label = result['label']  # Example: '4 stars'
        score = result['score']

        # Convert to Sentiment
        if label in ['1 star', '2 stars']:
            sentiment = "NEGATIVE"
            color = "red"
        elif label == '3 stars':
            sentiment = "NEUTRAL"
            color = "orange"
        else:
            sentiment = "POSITIVE"
            color = "limegreen"

        # Show result
        st.markdown(f"<h2 style='color:{color};'>Sentiment: {sentiment}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:deepskyblue;'>Confidence: {score:.2f}</h4>", unsafe_allow_html=True)
