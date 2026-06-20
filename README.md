🧠 AI-Generated vs Human-Written Text Detection using NLP

📌 Project Overview

With the rapid advancement of Large Language Models (LLMs) such as ChatGPT, Gemini, Claude, and other generative AI systems, distinguishing AI-generated content from human-written text has become an important challenge in academia, publishing, education, and content verification.

This project presents a complete AI Text Detection System that combines:

* Natural Language Processing (NLP)
* Machine Learning (ML)
* Deep Learning (RoBERTa)
* Ensemble Learning

The system classifies text into:

| Label | Meaning       |
| ----- | ------------- |
| 0     | Human Written |
| 1     | AI Generated  |

---

🎯 Key Features

* Complete NLP Pipeline
* Text Cleaning & Preprocessing
* Hybrid Feature Engineering
* TF-IDF Features
* Stylometric Features
* POS-based Linguistic Features
* Semantic Embeddings (SBERT)
* Logistic Regression
* XGBoost
* RoBERTa Transformer Model
* Ensemble Learning
* Streamlit Deployment
* Real-Time Prediction

---

📚 Research Foundation

This project is inspired by research conducted on AI-generated text detection.

### Research Paper

Research Paper Link:
https://drive.google.com/drive/folders/1dELjl1SUJa16CEklBh8yKLnPaFNmKM2i?usp=drive_link

### Dataset Source

Dataset Name:
CHEAT Dataset (IEEE Abstract Dataset)

Dataset Link:
https://github.com/botianzhe/CHEAT

### Dataset Usage Notice

The dataset used in this project is obtained from the original research source.

Due to licensing restrictions, copyright considerations, and repository size limitations, the dataset is NOT included in this GitHub repository.

Users are requested to download the dataset directly from the original source.

---

📊 Dataset Information

### Files Used

| File Name                    | Label |
| ---------------------------- | ----- |
| ieee-init.xlsx               | Human |
| ieee-chatgpt-generation.xlsx | AI    |
| ieee-chatgpt-polish.xlsx     | AI    |
| ieee-chatgpt-fusion.xlsx     | AI    |

### Final Dataset Statistics

* Total Samples: ~50,000
* Human Samples: ~15,000
* AI Samples: ~35,000

---

⚙️ Complete Project Pipeline

Raw Dataset

↓

Text Preprocessing

↓

Data Cleaning

↓

Balanced Train-Test Split

↓

Hybrid Feature Extraction

↓

Machine Learning Models

↓

RoBERTa Transformer

↓

Ensemble Learning

↓

Evaluation

↓

Deployment

---

🧹 Text Preprocessing

#File

preprocessing/preprocess.py

### Operations Performed

* Lowercasing
* URL Removal
* Special Character Removal
* Number Removal
* Tokenization
* Stopword Removal
* Lemmatization

# Output

data/final_preprocessed_data.csv

Columns:

* clean_text
* label

---

🔀 Dataset Splitting

# File

models/data_split.py

# Functionality

* Removes duplicate records
* Removes empty text samples
* Balances class distribution
* Prevents data leakage
* Creates train-test split

# Output

* train_data.csv
* test_data.csv

# Split Ratio

* Training: 80%
* Testing: 20%

---

🧠 Hybrid Feature Engineering

# File

features/hybrid_features_from_split.py

# Feature Categories

#### 1. TF-IDF Features

* Unigrams
* Bigrams
* Maximum Features: 5000

#### 2. Stylometric Features

* Average Word Length
* Average Sentence Length
* Stopword Ratio
* Readability Metrics

#### 3. POS Features

* Noun Ratio
* Verb Ratio
* Adjective Ratio
* Adverb Ratio

#### 4. Semantic Features

Model:
all-MiniLM-L6-v2

Embedding Dimension:
384

# Final Feature Vector

5395 Features Per Sample

---

🤖 Machine Learning Models

# File

models/train_models.py

# Models Implemented

1. Logistic Regression
2. XGBoost
3. Stacking Ensemble

# Best Traditional ML Model

Logistic Regression

Accuracy:
~86%

---

🧠 Deep Learning Model

# File

models/train_roberta.py

# Transformer Model

Model:
roberta-base

Training Configuration:

* Epochs: 3
* Max Sequence Length: 256
* GPU Enabled Training

---

🔥 Ensemble Model

Final Prediction Formula

0.7 × RoBERTa + 0.3 × Logistic Regression

# Why Ensemble?

RoBERTa captures:

* Context
* Semantics
* Deep Language Understanding

Logistic Regression captures:

* Statistical Writing Patterns
* Vocabulary Distribution

Combining both improves overall performance.

---

📊 Performance Evaluation

# File

evaluation/evaluate.py

# Metrics Used

* Accuracy
* Precision
* Recall
* F1 Score

# Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~86%     |
| RoBERTa             | ~89%     |
| Ensemble Model      | ~90%     |

# Final Performance

Accuracy:
89–90%

F1 Score:
89–90%

---

🚀 Streamlit Deployment

# File

app.py

# Features

* Text Input Box
* AI/Human Classification
* Confidence Score
* Ensemble Prediction
* Real-Time Inference

# Run Application

```bash
streamlit run app.py
```

# Prediction Output

* 🤖 AI Generated
* 👤 Human Written
* Confidence Percentage

---

📦 Installation Guide

# Clone Repository

```bash
https://github.com/LeutnantMutig/-AI-vs-Human-Text-Detection
```

```bash
cd AI vs Human Text-Detection
```

# Create Virtual Environment

```bash
python -m venv venv
```

Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

🏋️ Training From Scratch

Step 1

```bash
python preprocessing/preprocess.py
```

Step 2

```bash
python models/data_split.py
```

Step 3

```bash
python features/hybrid_features_from_split.py
```

Step 4

```bash
python models/train_models.py
```

Step 5

```bash
python models/train_roberta.py
```

---

📥 Pretrained Models

Due to GitHub size limitations, pretrained RoBERTa models are not included in this repository.

Download Pretrained Models:

Google Drive:
https://drive.google.com/drive/folders/1ORZli3qaT7Oo8f3AN9XfE-dJ82crCiNN?usp=drive_link

After downloading, place the files inside:

models/saved/

---

📁 Project Structure

AI-Text-Detection/

├── data/

├── preprocessing/

├── features/

├── models/

├── evaluation/

├── screenshots/

├── app.py

├── requirements.txt

├── README.md

└── .gitignore

---

📸 Screenshots

Add application screenshots here.

Examples:

* Streamlit Interface

  screenshots\Streamlit Interface.png\

* Prediction Results

  screenshots\Prediction Results.png\

---

⚠️ Limitations

* Academic writing may occasionally be misclassified.
* Human and AI writing styles may overlap.
* Performance depends on dataset diversity.
* Newer LLMs may require retraining for optimal detection.

---

🏁 Conclusion

This project demonstrates a complete end-to-end NLP solution for AI-generated text detection.

The project integrates:

* NLP
* Feature Engineering
* Machine Learning
* Deep Learning
* Ensemble Learning
* Real-Time Deployment

The final ensemble system achieves approximately 90% accuracy and provides a practical solution for detecting AI-generated content in academic and research environments.

---

👨‍💻 Author

Chirag Pawar

B.Tech Computer Science Engineering (AI & ML)

GitHub:
https://github.com[/LeutnantMutig](https://github.com/LeutnantMutig)

LinkedIn:
www.linkedin.com/in/chiragpawar01

⭐ Support

If you found this project useful, consider giving it a star on GitHub.

⭐ Star the repository 🍴 Fork the project 📢 Share with others
