import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
from streamlit_lottie import st_lottie

# Load model and tokenizer
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Set device to CPU
device = torch.device("cpu")
model.to(device)

# Function to classify text sentiment
def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    score, label_id = torch.max(probs, dim=1)
    label = label_id.item() + 1  # Convert 0-4 to 1-5 stars
    return label, score.item()

# Load Lottie animation
def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Streamlit page config
st.set_page_config(page_title="AI Sentiment Analyzer", layout="wide")
st.markdown("<h1 style='text-align: center; color: red;'> AI Text Sentiment Analyzer</h1>", unsafe_allow_html=True)

# Display Lottie animation
lottie_animation = load_lottie("animation.json")
st_lottie(lottie_animation, speed=1, reverse=False, loop=True, quality="high", height=250)

# Text input box
user_input = st.text_area("✏️ Enter your sentence:", height=100)

# Button to analyze
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence!")
    else:
        label, confidence = classify_text(user_input)

        # Determine sentiment
        if label in [1, 2]:
            sentiment = "NEGATIVE"
            color = "red"
        elif label == 3:
            sentiment = "NEUTRAL"
            color = "orange"
        else:
            sentiment = "POSITIVE"
            color = "limegreen"

        # Display results
        st.markdown(f"<h2 style='color:{color};'>Sentiment: {sentiment}</h2>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:deepskyblue;'>Confidence: {confidence:.2f}</h4>", unsafe_allow_html=True)
