import pandas as pd
import numpy as np
import re
import os
import joblib

import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from sentence_transformers import SentenceTransformer
import textstat

from scipy import sparse

# ===============================
# STEP 1: LOAD SPLIT DATA
# ===============================

train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv("data/test_data.csv")

X_train_text = train_df["clean_text"].astype(str)
y_train = train_df["label"].values

X_test_text = test_df["clean_text"].astype(str)
y_test = test_df["label"].values

print("Train size:", len(X_train_text))
print("Test size:", len(X_test_text))

# ===============================
# STEP 2: TF-IDF (BALANCED 🔥)
# ===============================

tfidf_vectorizer = TfidfVectorizer(
    ngram_range=(1,2),     # 🔥 REDUCED (important)
    max_features=3000      # 🔥 KEY FIX
)

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

print("TF-IDF done")

# ===============================
# STEP 3: STYLOMETRIC FEATURES
# ===============================

stop_words = set(stopwords.words("english"))

def extract_stylometric_features(text):
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    if len(words) == 0:
        return [0]*7

    avg_word_len = np.mean([len(w) for w in words])
    avg_sent_len = len(words) / max(len(sentences), 1)
    stopword_ratio = len([w for w in words if w.lower() in stop_words]) / len(words)
    punctuation_count = len(re.findall(r"[.,;:!?]", text))
    passive_voice_count = len(re.findall(r"\b(was|were|is|are|been|being)\b\s+\w+ed", text.lower()))
    flesch_score = textstat.flesch_reading_ease(text)
    gunning_fog = textstat.gunning_fog(text)

    return [
        avg_word_len,
        avg_sent_len,
        stopword_ratio,
        punctuation_count,
        passive_voice_count,
        flesch_score,
        gunning_fog
    ]

X_train_stylo = np.array([extract_stylometric_features(t) for t in X_train_text])
X_test_stylo = np.array([extract_stylometric_features(t) for t in X_test_text])

# ===============================
# STEP 4: POS FEATURES
# ===============================

POS_TAGS = ["NN", "JJ", "VB", "RB"]

def extract_pos_features(text):
    words = word_tokenize(text)
    if len(words) == 0:
        return [0]*4

    tagged = pos_tag(words)
    counts = {tag:0 for tag in POS_TAGS}

    for _, tag in tagged:
        if tag in counts:
            counts[tag] += 1

    total = len(words)
    return [
        counts["NN"]/total,
        counts["JJ"]/total,
        counts["VB"]/total,
        counts["RB"]/total
    ]

X_train_pos = np.array([extract_pos_features(t) for t in X_train_text])
X_test_pos = np.array([extract_pos_features(t) for t in X_test_text])

# ===============================
# STEP 5: SBERT (STRONG SIGNAL 🔥)
# ===============================

sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

X_train_sem = sbert_model.encode(
    X_train_text.tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

X_test_sem = sbert_model.encode(
    X_test_text.tolist(),
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True
)

# 🔥 NORMALIZE SBERT (IMPORTANT BOOST)
scaler_sem = StandardScaler()
X_train_sem = scaler_sem.fit_transform(X_train_sem)
X_test_sem = scaler_sem.transform(X_test_sem)

# ===============================
# STEP 6: NORMALIZATION
# ===============================

scaler_stylo = StandardScaler()
scaler_pos = StandardScaler()

X_train_stylo = scaler_stylo.fit_transform(X_train_stylo)
X_test_stylo = scaler_stylo.transform(X_test_stylo)

X_train_pos = scaler_pos.fit_transform(X_train_pos)
X_test_pos = scaler_pos.transform(X_test_pos)

# ===============================
# STEP 7: COMBINE (BALANCED 🔥)
# ===============================

X_train = sparse.hstack([
    X_train_tfidf,
    sparse.csr_matrix(X_train_stylo),
    sparse.csr_matrix(X_train_pos),
    sparse.csr_matrix(X_train_sem)
]).tocsr()

X_test = sparse.hstack([
    X_test_tfidf,
    sparse.csr_matrix(X_test_stylo),
    sparse.csr_matrix(X_test_pos),
    sparse.csr_matrix(X_test_sem)
]).tocsr()

print("Final train shape:", X_train.shape)
print("Final test shape:", X_test.shape)

# ===============================
# STEP 8: SAVE
# ===============================

os.makedirs("features/saved", exist_ok=True)

sparse.save_npz("features/saved/X_train.npz", X_train)
sparse.save_npz("features/saved/X_test.npz", X_test)

np.save("features/saved/y_train.npy", y_train)
np.save("features/saved/y_test.npy", y_test)

joblib.dump(tfidf_vectorizer, "features/saved/tfidf.pkl")
joblib.dump(scaler_stylo, "features/saved/stylo_scaler.pkl")
joblib.dump(scaler_pos, "features/saved/pos_scaler.pkl")
joblib.dump(scaler_sem, "features/saved/sem_scaler.pkl")

print("🔥 Hybrid features extracted (FINAL OPTIMIZED)")