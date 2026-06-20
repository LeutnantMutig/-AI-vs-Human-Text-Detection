import numpy as np
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ===============================
# DEVICE SETUP (GPU)
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===============================
# LOAD TEST DATA
# ===============================

test_df = pd.read_csv("data/test_data.csv")

texts = test_df["clean_text"].astype(str).tolist()
labels = test_df["label"].astype(int).values

print("Test size:", len(texts))


# ===============================
# LOAD TRAIN DATA (FOR LR)
# ===============================

train_df = pd.read_csv("data/train_data.csv")

train_texts = train_df["clean_text"].astype(str).tolist()
train_labels = train_df["label"].astype(int).values


# ===============================
# LOAD RoBERTa MODEL
# ===============================

print("\n🚀 Loading RoBERTa...")

tokenizer = AutoTokenizer.from_pretrained("models/saved/roberta_model")
model = AutoModelForSequenceClassification.from_pretrained("models/saved/roberta_model")

model.to(device)
model.eval()


# ===============================
# TRAIN LOGISTIC REGRESSION
# ===============================

print("🚀 Training Logistic Regression...")

vectorizer = TfidfVectorizer(max_features=5000)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(texts)

lr_model = LogisticRegression(max_iter=1000, n_jobs=-1)
lr_model.fit(X_train, train_labels)

lr_probs = lr_model.predict_proba(X_test)[:, 1]


# ===============================
# RoBERTa PREDICTION (GPU + BATCH)
# ===============================

print("🚀 Running RoBERTa (GPU BATCH MODE)...")

batch_size = 32
roberta_probs = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i + batch_size]

    inputs = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    )

    # Move to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[:, 1]
    roberta_probs.extend(probs.detach().cpu().numpy())

roberta_probs = np.array(roberta_probs)


# ===============================
# ENSEMBLE
# ===============================

print("🚀 Applying Ensemble...")

final_probs = (0.7 * roberta_probs) + (0.3 * lr_probs)

threshold = 0.66
final_preds = (final_probs >= threshold).astype(int)


# ===============================
# FINAL METRICS
# ===============================

print("\n🔥 FINAL ENSEMBLE RESULTS")

accuracy = accuracy_score(labels, final_preds)
precision = precision_score(labels, final_preds)
recall = recall_score(labels, final_preds)
f1 = f1_score(labels, final_preds)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")