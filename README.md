# TikTok Sentiment Analysis & Content Summarization - Big Data Project

## Project Overview  
This project focuses on analyzing user sentiments toward sunscreen products on TikTok using Big Data technologies and Natural Language Processing. The repository contains datasets, trained machine learning models, and Jupyter notebooks for sentiment analysis and video content summarization.

---

## Repository Structure  

### 1. **Data Folder**  
The `data` folder contains raw and processed datasets used in the analysis.  

- **Raw Data**
  The raw data is sourced from TikTok using Apify scrapers.  
  - `comment_data/`: Contains JSON files of raw comments scraped from TikTok videos.  
  - `video_data/`: Includes JSON files with metadata of TikTok videos (e.g., views, likes, hashtags).  
  - `teencode4.txt`: A list of Vietnamese internet slang and abbreviations for text preprocessing.  
  - `vietnamese_stopwords.txt`: A list of Vietnamese stopwords used to filter non-informative terms during NLP tasks.  

- **Processed Data**  
  - `all_comment_labeled.tsv`: Labeled comments (positive, neutral, negative) for training sentiment analysis models.  
  - `train.tsv`, `val.tsv`, `test.tsv`: Split datasets for training, validation, and testing.  
  - `unlabeled_features.tsv`: Unlabeled comment embeddings extracted using PhoBERT for pseudo-labeling.  

---

### 2. **Model Folder**  
The `model` folder stores pre-trained machine learning models for sentiment classification:  
- `LogisticRegression_best_model.pkl`: Optimized Logistic Regression model.  
- `mlp_model.pkl`: Multilayer Perceptron (MLP) model.  
- `RandomForestClassifier_best_model.pkl`: Random Forest classifier.  
- `SVC_best_model.pkl`: Support Vector Machine (SVM) model.  

These models were trained on labeled comment data and can be used to predict sentiments on new TikTok comments.  

---

### 3. **Code**  

#### **Gemini_summarize.ipynb**  
This notebook automates the summarization of TikTok video content. It includes:  
- **Dependencies**:  
  - `SpeechRecognition`: Converts audio to text.  
  - `yt-dlp`: Downloads TikTok videos.  
  - `ffmpeg`: Extracts audio from videos.  
  - `pyspark`: Processes data in parallel.  

**Functionality**:  
1. Extracts audio from TikTok videos using URLs.  
2. Converts speech to text.  
3. Summarizes content using Gemini API.  

---

#### **TikTok_sentiment.ipynb**  
This notebook performs sentiment analysis on TikTok comments. Key steps include:  
- **Dependencies**:  
  - `pyspark`: Handles large-scale data processing.  
  - `pyvi` and `underthesea`: Vietnamese NLP tools for text preprocessing.  
  - `emoji`, `unidecode`: Clean and normalize text.  
  - `transformers`: Uses PhoBERT for text embedding.  
  - `tensorflow`: Trains deep learning models.  

**Workflow**:  
1. Preprocesses raw comments (tokenization, stopword removal).  
2. Extracts features using PhoBERT.  
3. Trains and evaluates machine learning models.  
4. Implements pseudo-labeling to expand the training dataset.  

---

## Setup Instructions  

### Install Dependencies  
```bash
# For Gemini_summarize.ipynb
pip install SpeechRecognition yt-dlp ffmpeg-python pyspark

# For TikTok_sentiment.ipynb
pip install pyspark pyvi underthesea emoji unidecode transformers tensorflow
```

### Running the Notebooks  
1. Ensure all dependencies are installed.  
2. Download the datasets and place them in the `data` folder.  
3. Open the notebooks in Jupyter or Google Colab.  
4. Update file paths in the notebooks to match your local directory structure.

---
## Key Research Findings  

### 1. **Data Analysis**  
- **Video Metadata**: Analyzed 309 TikTok videos, identifying trends in engagement (e.g., average views, likes, shares). Ads had 5.6× higher views but 2.3× lower engagement rates compared to organic content.  
- **Top Hashtags**: `#kemchongnang` (sunscreen) and `#skincare` dominated, reflecting user interest in beauty and skincare.  
- **Comment Insights**: Processed 13,007 comments, with 46.3% neutral, 28% positive, and 25.6% negative sentiments.  

### 2. **Sentiment Analysis**  
- **Model Performance**:  
  - **Logistic Regression** achieved the highest accuracy (**84%**) in classifying sentiments.  
  - **Pseudo-Labeling** improved model robustness by leveraging unlabeled data, balancing class distribution.  
- **Dominant Themes**: Users prioritized product suitability for skin types (e.g., oily, sensitive) and expressed concerns about counterfeit products.  

### 3. **Video Summarization**  
- **Efficiency**: A hybrid pipeline combining `yt-dlp`, `SpeechRecognition`, and **Gemini API** successfully summarized video content.  
- **Speed Comparison**: ThreadPoolExecutor (5 threads) processed 100 videos **2.4× faster** than PySpark (2 partitions).  

### 4. **Technical Contributions**  
- **NLP Pipeline**: Integrated PhoBERT embeddings for Vietnamese text, achieving state-of-the-art results in sentiment classification.  
- **Open Datasets**: Published labeled comment data (`all_comment_labeled.tsv`) and preprocessing scripts for reproducibility.  

---

## Contributors
- Nguyễn Mai Hồng Trâm
- Trần Vọng Triển
- Nguyễn Ngọc Thúy Anh
- Đỗ Ngọc Phương Anh
- Huỳnh Minh Phương
---
## Acknowledgement
We would like to sincerely thank Dr. Nguyễn Mạnh Tuấn for his dedicated instruction and valuable feedback throughout the Big Data and Applications course.
We also appreciate the constructive comments and suggestions from other student groups in the Big Data and Applications course (UEH Semester 1, 2025), which helped us identify mistakes and find appropriate ways to improve our project.
