import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

@st.cache_resource
def load_model():
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

# Sentiment analysis function
def analyze_sentiment(text, tokenizer, model):
    # Tokenize input
    encoded_input = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**encoded_input)
        scores = outputs.logits.softmax(dim=-1)
    
    # Labels mapping
    labels = ['Negative', 'Neutral', 'Positive']
    sentiment = labels[scores.argmax()]
    confidence = scores.max().item()
    
    return sentiment, confidence

# Streamlit app
def main():
    # Load model
    tokenizer, model = load_model()
    
    # UI Setup
    st.title("Multilingual Social Media Sentiment Analysis")
    st.write("Enter text in any language to analyze its sentiment")
    
    # Input text area
    user_input = st.text_area("Enter your text here", height=150)
    
    # Analyze button
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            with st.spinner("Analyzing..."):
                sentiment, confidence = analyze_sentiment(user_input, tokenizer, model)
                
                # Display results
                st.subheader("Results:")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"Sentiment: **{sentiment}**")
                with col2:
                    st.write(f"Confidence: **{confidence:.2%}**")
                
                # Sentiment emoji
                emoji = "😞" if sentiment == "Negative" else "😐" if sentiment == "Neutral" else "😊"
                st.write(f"Sentiment Visualization: {emoji}")
        else:
            st.warning("Please enter some text to analyze!")

    # Example texts
    st.subheader("Try these examples:")
    examples = {
        "English": "I love this beautiful day!",
        "Spanish": "¡Odio este clima horrible!",
        "French": "C'est une journée magnifique",
        "German": "Ich hasse diesen Tag"
    }
    
    for lang, text in examples.items():
        if st.button(f"Try {lang} example"):
            sentiment, confidence = analyze_sentiment(text, tokenizer, model)
            st.write(f"Text: '{text}'")
            st.write(f"Sentiment: {sentiment} ({confidence:.2%})")

if __name__ == "__main__":
    main()