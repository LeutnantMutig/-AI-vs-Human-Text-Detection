<<<<<<< HEAD
# 🧠 AI-Generated vs Human-Written Text Detection using NLP

## 📌 Project Overview
=======
🧠 AI-Generated vs Human-Written Text Detection using NLP

📌 Project Overview
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

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

<<<<<<< HEAD
## 🎯 Key Features
=======
🎯 Key Features
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

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

<<<<<<< HEAD
# 📚 Research Foundation
=======
📚 Research Foundation
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

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

<<<<<<< HEAD
# 📊 Dataset Information
=======
📊 Dataset Information
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

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

<<<<<<< HEAD
# ⚙️ Complete Project Pipeline
=======
⚙️ Complete Project Pipeline
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

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

<<<<<<< HEAD
# 🧹 Text Preprocessing

### File
=======
🧹 Text Preprocessing

#File
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

preprocessing/preprocess.py

### Operations Performed

* Lowercasing
* URL Removal
* Special Character Removal
* Number Removal
* Tokenization
* Stopword Removal
* Lemmatization

<<<<<<< HEAD
### Output
=======
# Output
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

data/final_preprocessed_data.csv

Columns:

* clean_text
* label

---

<<<<<<< HEAD
# 🔀 Dataset Splitting

### File

models/data_split.py

### Functionality
=======
🔀 Dataset Splitting

# File

models/data_split.py

# Functionality
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

* Removes duplicate records
* Removes empty text samples
* Balances class distribution
* Prevents data leakage
* Creates train-test split

<<<<<<< HEAD
### Output
=======
# Output
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

* train_data.csv
* test_data.csv

<<<<<<< HEAD
### Split Ratio
=======
# Split Ratio
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

* Training: 80%
* Testing: 20%

---

<<<<<<< HEAD
# 🧠 Hybrid Feature Engineering

### File

features/hybrid_features_from_split.py

### Feature Categories
=======
🧠 Hybrid Feature Engineering

# File

features/hybrid_features_from_split.py

# Feature Categories
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

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

<<<<<<< HEAD
### Final Feature Vector
=======
# Final Feature Vector
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

5395 Features Per Sample

---

<<<<<<< HEAD
# 🤖 Machine Learning Models

### File

models/train_models.py

### Models Implemented
=======
🤖 Machine Learning Models

# File

models/train_models.py

# Models Implemented
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

1. Logistic Regression
2. XGBoost
3. Stacking Ensemble

<<<<<<< HEAD
### Best Traditional ML Model
=======
# Best Traditional ML Model
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

Logistic Regression

Accuracy:
~86%

---

<<<<<<< HEAD
# 🧠 Deep Learning Model

### File

models/train_roberta.py

### Transformer Model
=======
🧠 Deep Learning Model

# File

models/train_roberta.py

# Transformer Model
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

Model:
roberta-base

Training Configuration:

* Epochs: 3
* Max Sequence Length: 256
* GPU Enabled Training

---

<<<<<<< HEAD
# 🔥 Ensemble Model
=======
🔥 Ensemble Model
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

Final Prediction Formula

0.7 × RoBERTa + 0.3 × Logistic Regression

<<<<<<< HEAD
### Why Ensemble?
=======
# Why Ensemble?
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

RoBERTa captures:

* Context
* Semantics
* Deep Language Understanding

Logistic Regression captures:

* Statistical Writing Patterns
* Vocabulary Distribution

Combining both improves overall performance.

---

<<<<<<< HEAD
# 📊 Performance Evaluation

### File

evaluation/evaluate.py

### Metrics Used
=======
📊 Performance Evaluation

# File

evaluation/evaluate.py

# Metrics Used
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

* Accuracy
* Precision
* Recall
* F1 Score

<<<<<<< HEAD
### Results
=======
# Results
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~86%     |
| RoBERTa             | ~89%     |
| Ensemble Model      | ~90%     |

<<<<<<< HEAD
### Final Performance
=======
# Final Performance
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

Accuracy:
89–90%

F1 Score:
89–90%

---

<<<<<<< HEAD
# 🚀 Streamlit Deployment

### File

app.py

### Features
=======
🚀 Streamlit Deployment

# File

app.py

# Features
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

* Text Input Box
* AI/Human Classification
* Confidence Score
* Ensemble Prediction
* Real-Time Inference

<<<<<<< HEAD
### Run Application
=======
# Run Application
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

```bash
streamlit run app.py
```

<<<<<<< HEAD
### Prediction Output
=======
# Prediction Output
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

* 🤖 AI Generated
* 👤 Human Written
* Confidence Percentage

---

<<<<<<< HEAD
# 📦 Installation Guide

## Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
=======
📦 Installation Guide

# Clone Repository

```bash
https://github.com/LeutnantMutig/-AI-vs-Human-Text-Detection
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6
```

```bash
cd AI vs Human Text-Detection
```

<<<<<<< HEAD
## Create Virtual Environment
=======
# Create Virtual Environment
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

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

<<<<<<< HEAD
## Install Dependencies
=======
# Install Dependencies
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

```bash
pip install -r requirements.txt
```

---

<<<<<<< HEAD
# 🏋️ Training From Scratch
=======
🏋️ Training From Scratch
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

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

<<<<<<< HEAD
# 📥 Pretrained Models
=======
📥 Pretrained Models
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

Due to GitHub size limitations, pretrained RoBERTa models are not included in this repository.

Download Pretrained Models:

Google Drive:
https://drive.google.com/drive/folders/1ORZli3qaT7Oo8f3AN9XfE-dJ82crCiNN?usp=drive_link

After downloading, place the files inside:

models/saved/

---

<<<<<<< HEAD
# 📁 Project Structure
=======
📁 Project Structure
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

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

<<<<<<< HEAD
# 📸 Screenshots
=======
📸 Screenshots
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

Add application screenshots here.

Examples:

* Streamlit Interface

  screenshots\Streamlit Interface.png\

* Prediction Results

  screenshots\Prediction Results.png\

---

<<<<<<< HEAD
# ⚠️ Limitations
=======
⚠️ Limitations
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

* Academic writing may occasionally be misclassified.
* Human and AI writing styles may overlap.
* Performance depends on dataset diversity.
* Newer LLMs may require retraining for optimal detection.

---

<<<<<<< HEAD
# 🏁 Conclusion
=======
🏁 Conclusion
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

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

<<<<<<< HEAD
## 👨‍💻 Author
=======
👨‍💻 Author
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

Chirag Pawar

B.Tech Computer Science Engineering (AI & ML)

GitHub:
<<<<<<< HEAD
 [https://github.com[/LeutnantMutig](https://github.com[/LeutnantMutig%5D%28https://github.com/LeutnantMutig%29)

LinkedIn:
[www.linkedin.com/in/chiragpawar01](http://www.linkedin.com/in/chiragpawar01)

# ⭐ Support
=======
https://github.com[/LeutnantMutig](https://github.com/LeutnantMutig)

LinkedIn:
www.linkedin.com/in/chiragpawar01

⭐ Support
>>>>>>> d1fd0ea9ce3225b8e63b3b0ade2fb97c24c1b2d6

If you found this project useful, consider giving it a star on GitHub.

⭐ Star the repository 🍴 Fork the project 📢 Share with others
