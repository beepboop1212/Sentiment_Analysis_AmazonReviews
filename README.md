---

## 🛍️ Sentiment Analysis of Amazon Reviews

This notebook demonstrates how to build a **sentiment analysis** pipeline to classify Amazon product reviews as **positive** or **negative**. It uses machine learning/NLP techniques to understand customer feedback and gauge overall satisfaction.

---

### 📌 Project Overview

- **Dataset**: Amazon product reviews (binary sentiment: Positive / Negative)
- **Task**: Text classification
- **Goal**: Predict sentiment of a review using NLP models
- **Approach**: Preprocessing → Vectorization → Model Training → Evaluation

---


## 📊 Dataset Overview

- 📁 Source: `https://www.kaggle.com/datasets/eswarchandt/amazon-music-reviews` (from Kaggle)
- This file has reviewer ID , User ID, Reviewer Name, Reviewer text, helpful, Summary(obtained from Reviewer text),Overall Rating on a scale 5, Review time
Description of columns in the file:

reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
asin - ID of the product, e.g. 0000013714
reviewerName - name of the reviewer
helpful - helpfulness rating of the review, e.g. 2/3
reviewText - text of the review
overall - rating of the product
summary - summary of the review
unixReviewTime - time of the review (unix time)
reviewTime - time of the review (raw)

---


### 📂 Notebook Contents

| Section | Description |
|--------|-------------|
| 📊 Data Loading | Load and inspect Amazon review data |
| 🧼 Text Cleaning | Remove noise: punctuation, stopwords, HTML, etc. |
| 🔢 Feature Extraction | Convert text into numerical features using TF-IDF or CountVectorizer |
| 🤖 Model Training | Train classifiers such as Logistic Regression, Naive Bayes, or SVM |
| 📈 Evaluation | Accuracy, precision, recall, F1-score, confusion matrix |
| 🔮 Inference | Test model on custom input reviews |
| 📊 Visualization | Word clouds or bar plots of most common words |

---

### 🛠️ Requirements

```bash
pandas
numpy
scikit-learn
nltk
matplotlib
seaborn
wordcloud
```

Install with:

```bash
pip install pandas numpy scikit-learn nltk matplotlib seaborn wordcloud
```

---

### 🚀 Sample Inference

```python
review = "This product exceeded my expectations!"
prediction = model.predict(vectorizer.transform([review]))
print("Sentiment:", "Positive" if prediction[0] == 1 else "Negative")
```

---

### 📌 Notes

- You can extend this model using:
  - Deep learning models (LSTM, BERT, etc.)
  - Multiclass sentiment (e.g., star ratings)
  - Topic modeling or aspect-based sentiment analysis
- This is a great base for e-commerce analytics or review summarization.

---
