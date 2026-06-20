import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

# --------------------------------------------------
# NLP SETUP
# --------------------------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Robust text preprocessing function
    SAFE for NaN, int, float, empty values
    """

    if pd.isna(text):
        return ""

    # Force conversion to string (CRITICAL FIX)
    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove special characters & numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword removal
    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def preprocess_dataset(file_path, label):
    """
    Loads dataset, preprocesses ONLY abstract column,
    assigns label, returns clean dataframe
    """

    df = pd.read_excel(file_path)

    # ✅ EXPLICITLY select correct column
    if "abstract" not in df.columns:
        raise ValueError(f"'abstract' column not found in {file_path}")

    tqdm.pandas(desc=f"Processing {file_path.split('/')[-1]}")

    df["clean_text"] = df["abstract"].progress_apply(clean_text)
    df["label"] = label

    return df[["clean_text", "label"]]


if __name__ == "__main__":

    print("Starting preprocessing...")

    # -------------------------------
    # LOAD ALL FOUR DATASETS
    # -------------------------------

    human_df = preprocess_dataset(
        r"E:\SEM 8\AI-Text-Detection\data\ieee-init.xlsx",
        label=0
    )

    ai_gen_df = preprocess_dataset(
        r"E:\SEM 8\AI-Text-Detection\data\ieee-chatgpt-generation.xlsx",
        label=1
    )

    ai_polish_df = preprocess_dataset(
        r"E:\SEM 8\AI-Text-Detection\data\ieee-chatgpt-polish.xlsx",
        label=1
    )

    ai_fusion_df = preprocess_dataset(
        r"E:\SEM 8\AI-Text-Detection\data\ieee-chatgpt-fusion.xlsx",
        label=1
    )

    # -------------------------------
    # MERGE ALL DATA
    # -------------------------------

    final_df = pd.concat(
        [human_df, ai_gen_df, ai_polish_df, ai_fusion_df],
        ignore_index=True
    )

    # Shuffle data
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save final dataset
    final_df.to_csv(
        r"E:\SEM 8\AI-Text-Detection\data\final_preprocessed_data.csv",
        index=False
    )

    print("✅ Preprocessing completed successfully!")
    print("📁 Saved: final_preprocessed_data.csv")
