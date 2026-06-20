import streamlit as st
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# ===============================
# LOAD MODELS
# ===============================

@st.cache_resource
def load_roberta():
    model = AutoModelForSequenceClassification.from_pretrained("models/saved/roberta_model")
    tokenizer = AutoTokenizer.from_pretrained("models/saved/roberta_model")
    return model, tokenizer


model, tokenizer = load_roberta()
model.eval()


@st.cache_resource
def load_lr():
    df = pd.read_csv("data/train_data.csv")

    texts = df["clean_text"].astype(str).tolist()
    labels = df["label"].astype(int).tolist()

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X, labels)

    return lr, vectorizer

lr_model, vectorizer = load_lr()


# ===============================
# PREDICTION FUNCTION
# ===============================

def predict(text):
    # RoBERTa Prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)

    with torch.no_grad():
        outputs = model(**inputs)

    roberta_probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    roberta_ai = roberta_probs[1].item()

    # Logistic Regression Prediction
    tfidf = vectorizer.transform([text])
    lr_prob = lr_model.predict_proba(tfidf)[0][1]

    # Ensemble
    final_prob = (0.7 * roberta_ai) + (0.3 * lr_prob)

    threshold = 0.75

    if final_prob >= threshold:
        label = "🤖 AI GENERATED"
        binary = 1
    else:
        label = "🧑 HUMAN WRITTEN"
        binary = 0

    # if final_prob >= 0.75:
    #     label = "🤖 AI GENERATED"
    #     binary = 1

    # elif final_prob <= 0.35:
    #     label = "🧑 HUMAN WRITTEN"
    #     binary = 0

    # else:
    #     label = "⚠️ UNCERTAIN"
    #     binary = -1


    return {
        "label": label,
        "binary": binary,
        "ai_prob": final_prob,
        "human_prob": 1 - final_prob
    }


# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(page_title="AI Text Detector", layout="centered")

st.title("🔥 AI vs Human Text Detector")
st.write("Paste text below to check if it's AI-generated or Human-written")

user_input = st.text_area("Enter Text Here", height=200)


if st.button("🔍 Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text")

    else:
        result = predict(user_input)

        # RESULT
        st.subheader("📊 Result")
        st.success(result["label"])

        # BINARY OUTPUT
        st.subheader("🔢 Binary Output")
        st.info(f"{result['binary']}  (1 = AI, 0 = Human)")

        # CONFIDENCE
        st.subheader("📈 Confidence Scores")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("🤖 AI Probability", f"{result['ai_prob']*100:.2f}%")

        with col2:
            st.metric("🧑 Human Probability", f"{result['human_prob']*100:.2f}%")

        # PROGRESS BAR
        st.progress(float(result["ai_prob"]))

        # EXTRA INFO
        st.caption("Threshold used: 0.75 (optimized for best F1 score)")