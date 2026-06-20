# =====================================================
# DATA SPLIT PIPELINE (FINAL CLEAN + NO LEAKAGE)
# =====================================================

import pandas as pd
from sklearn.model_selection import train_test_split

# ===============================
# STEP 1: LOAD DATA
# ===============================

df = pd.read_csv("data/final_preprocessed_data.csv")

print("📊 Original Distribution:")
print(df["label"].value_counts())

# ===============================
# STEP 2: REMOVE DUPLICATES (IMPORTANT FIX)
# ===============================

# 🔥 Remove duplicates based on clean_text ONLY
df = df.drop_duplicates(subset=["clean_text"]).reset_index(drop=True)

print("\n📊 After Removing Duplicates:")
print(df["label"].value_counts())

# ===============================
# STEP 3: REMOVE EMPTY TEXT
# ===============================

df = df[df["clean_text"].str.strip() != ""]

print("\n📊 After Removing Empty Text:")
print(df["label"].value_counts())

# ===============================
# STEP 4: BALANCE DATASET (SAFE)
# ===============================

df_human = df[df["label"] == 0]
df_ai = df[df["label"] == 1]

min_class = min(len(df_human), len(df_ai))

df_human_bal = df_human.sample(n=min_class, random_state=42)
df_ai_bal = df_ai.sample(n=min_class, random_state=42)

df_balanced = pd.concat([df_human_bal, df_ai_bal])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n📊 Balanced Distribution:")
print(df_balanced["label"].value_counts())

# ===============================
# STEP 5: TRAIN-TEST SPLIT
# ===============================

train_df, test_df = train_test_split(
    df_balanced,
    test_size=0.2,
    stratify=df_balanced["label"],
    random_state=42
)

# ===============================
# STEP 6: REMOVE TRAIN-TEST OVERLAP (CRITICAL FIX)
# ===============================

train_texts = set(train_df["clean_text"])

# Remove overlapping rows from test set
test_df = test_df[~test_df["clean_text"].isin(train_texts)].reset_index(drop=True)

# ===============================
# STEP 7: FINAL SAFETY CHECK
# ===============================

overlap = set(train_df["clean_text"]).intersection(set(test_df["clean_text"]))

print("\n📊 Train Size:", len(train_df))
print("📊 Test Size:", len(test_df))
print("🚨 Train-Test Overlap Rows:", len(overlap))

# ===============================
# STEP 8: SAVE FILES
# ===============================

train_df.to_csv("data/train_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)

print("\n✅ Train & Test files saved successfully!")