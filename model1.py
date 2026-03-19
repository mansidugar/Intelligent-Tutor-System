# ================================
# INSTALL (run once in terminal)
# pip install pandas scikit-learn matplotlib seaborn
# ================================

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# ================================
# LOAD DATASETS
# ================================

questions = pd.read_csv(r"C:\Users\mansi\Downloads\SNLP MODEL\archive (1)\Questions.csv", encoding="latin1")
tags      = pd.read_csv(r"C:\Users\mansi\Downloads\SNLP MODEL\archive (1)\Tags.csv", encoding="latin1")

# ================================
# MERGE QUESTIONS + TAGS
# ================================

df = pd.merge(questions, tags, on="Id")

# ================================
# BUILD TEXT INPUT
# ================================

df["text"] = df["Title"].fillna("") + " " + df["Body"].fillna("")

# Remove HTML tags
df["text"] = df["text"].apply(lambda x: re.sub(r'<.*?>', ' ', x))

# Remove empty rows
df = df.dropna(subset=["text", "Tag"])

# ================================
# REMOVE RARE TAGS (IMPORTANT)
# ================================

tag_counts = df["Tag"].value_counts()
df = df[df["Tag"].isin(tag_counts[tag_counts > 1000].index)]

# ================================
# SPEED: USE SUBSET
# ================================

df = df.sample(60000, random_state=42)

# ================================
# FEATURES & LABELS
# ================================

X = df["text"]
y = df["Tag"]

# ================================
# TRAIN-TEST SPLIT
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================================
# MODEL (FASTER TF-IDF + NB)
# ================================

intent_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=3000,
        ngram_range=(1,1)
    )),
    ("nb", MultinomialNB(alpha=0.5))
])

# ================================
# TRAIN
# ================================

intent_model.fit(X_train, y_train)

# ================================
# PREDICT
# ================================

preds = intent_model.predict(X_test)

print("\n=== CLASSIFICATION REPORT ===\n")
print(classification_report(y_test, preds, zero_division=0))

# ================================
# TEST WITH CUSTOM QUERY
# ================================

query = ["why index out of range error python"]
print("\nPredicted Intent:", intent_model.predict(query))

# ================================
# VISUAL 1: CLASS DISTRIBUTION
# ================================

plt.figure(figsize=(10,5))
y.value_counts().head(15).plot(kind="bar")
plt.title("Top 15 Class Distribution")
plt.ylabel("Count")
plt.xlabel("Tag")
plt.tight_layout()
plt.show()

# ================================
# VISUAL 2: CONFUSION MATRIX (TOP CLASSES)
# ================================

top_labels = y_test.value_counts().nlargest(10).index

mask = y_test.isin(top_labels)
y_true_top = y_test[mask]
y_pred_top = preds[mask]

cm = confusion_matrix(y_true_top, y_pred_top, labels=top_labels)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=top_labels,
            yticklabels=top_labels)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Top 10 Classes)")
plt.tight_layout()
plt.show()

# ================================
# VISUAL 3: MODEL PERFORMANCE
# ================================

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, preds, average='weighted'
)

metrics = ['Precision', 'Recall', 'F1 Score']
values = [precision, recall, f1]

plt.figure(figsize=(6,4))
plt.bar(metrics, values)
plt.title("Model Performance")
plt.ylim(0,1)
plt.tight_layout()
plt.show()
