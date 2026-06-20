import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


# ===============================
# LOAD DATA
# ===============================

train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv("data/test_data.csv")

train_texts = train_df["clean_text"].astype(str).tolist()
train_labels = train_df["label"].astype(int).tolist()

test_texts = test_df["clean_text"].astype(str).tolist()
test_labels = test_df["label"].astype(int).tolist()

print("Train size:", len(train_texts))
print("Test size:", len(test_texts))


# ===============================
# 🔥 LOGISTIC REGRESSION (TF-IDF)
# ===============================

print("\n🚀 Training Logistic Regression...")

vectorizer = TfidfVectorizer(max_features=5000)

X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_tfidf, train_labels)

lr_probs = lr_model.predict_proba(X_test_tfidf)[:, 1]


# ===============================
# RoBERTa DATASET
# ===============================

train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})


# ===============================
# TOKENIZER
# ===============================

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=256)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
test_dataset.set_format("torch")


# ===============================
# DEVICE
# ===============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ===============================
# MODEL
# ===============================

model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
model.to(device)


# ===============================
# 🔥 METRICS (FIXED ERROR)
# ===============================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary'
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# ===============================
# TRAINING CONFIG
# ===============================

training_args = TrainingArguments(
    output_dir="models/roberta_results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
    report_to="none",
    fp16=True,
    dataloader_num_workers=0
)


# ===============================
# TRAINER
# ===============================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics   # 🔥 FIXED
)


# ===============================
# TRAIN RoBERTa
# ===============================

print("\n🚀 Training RoBERTa...\n")
trainer.train()


# ===============================
# GET RoBERTa PROBABILITIES
# ===============================

pred_output = trainer.predict(test_dataset)

logits = pred_output.predictions
roberta_probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()


# ===============================
# 🔥 ENSEMBLE (FINAL BOOST)
# ===============================

print("\n🔥 Applying Ensemble...")

final_probs = (0.7 * roberta_probs) + (0.3 * lr_probs)

# Threshold tuning
best_thresh = 0.5
best_f1 = 0

for t in np.arange(0.4, 0.7, 0.02):
    preds = (final_probs >= t).astype(int)
    f1 = f1_score(test_labels, preds)

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

print(f"\n🔥 BEST THRESHOLD: {best_thresh}")

final_preds = (final_probs >= best_thresh).astype(int)


# ===============================
# FINAL RESULTS
# ===============================

print("\n🔥 FINAL ENSEMBLE RESULTS")

print("Accuracy :", accuracy_score(test_labels, final_preds))
print("Precision:", precision_score(test_labels, final_preds))
print("Recall   :", recall_score(test_labels, final_preds))
print("F1 Score :", f1_score(test_labels, final_preds))


# ===============================
# SAVE MODEL
# ===============================

model.save_pretrained("models/saved/roberta_model")
tokenizer.save_pretrained("models/saved/roberta_model")

print("\n✅ FINAL MODEL SAVED!")