# ==========================================
# MODEL 2: CONCEPT GAP DETECTION
# Synthetic Realistic Tutoring Dataset (50–100 words each)
# Model: Logistic Regression
# ==========================================

import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# ==========================================
# STEP 1: CREATE REALISTIC DATASET
# ==========================================

understood_samples = [
    """Recursion is a programming technique where a function calls itself repeatedly until a base condition is satisfied. 
    I understand that each recursive call creates a new stack frame and the process continues until the stopping condition 
    prevents infinite looping. This helps solve problems like factorials, tree traversal, and divide and conquer algorithms 
    efficiently by breaking them into smaller subproblems.""",

    """A loop is used to repeat a block of code until a certain condition is met. I understand the difference between 
    for loops and while loops and how iteration helps in processing collections such as lists or arrays. Loops improve 
    efficiency and reduce code duplication when solving repetitive tasks like counting, searching, and traversing data.""",

    """Object oriented programming is based on classes and objects. I understand encapsulation, inheritance, and 
    polymorphism, and how they help structure programs into reusable and modular components. This improves code 
    maintainability and scalability in large software systems."""
]

not_understood_samples = [
    """I am confused about recursion because the function keeps calling itself and I cannot understand when it stops. 
    Sometimes my program runs forever and I don’t know why. The base condition is not clear to me and I struggle to 
    visualize how recursive calls return values back to the original function call.""",

    """Loops are confusing to me because I don’t understand how many times they should run. Sometimes my loop becomes 
    infinite and crashes the program. I am not sure when to use a for loop or a while loop and I often get errors while 
    trying to iterate over lists or arrays.""",

    """Object oriented programming concepts are difficult for me to understand. I get confused between classes and 
    objects and don’t know how inheritance works. Polymorphism and encapsulation seem complicated and I cannot apply 
    them properly in my code."""
]

# Generate 200 samples (balanced)
texts = []
labels = []

for _ in range(100):
    texts.append(random.choice(understood_samples))
    labels.append("understood")

for _ in range(100):
    texts.append(random.choice(not_understood_samples))
    labels.append("not_understood")

df = pd.DataFrame({"text": texts, "label": labels})

print("Dataset created:", df.shape)

# ==========================================
# STEP 2: TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# ==========================================
# STEP 3: MODEL PIPELINE
# ==========================================

model_gap = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
model_gap.fit(X_train, y_train)

# ==========================================
# STEP 4: EVALUATE
# ==========================================

preds = model_gap.predict(X_test)

print("\n=== CONCEPT GAP MODEL REPORT ===\n")
print(classification_report(y_test, preds))

# ==========================================
# STEP 5: TEST WITH NEW INPUT
# ==========================================

test_text = ["I do not understand recursion and my function keeps looping forever"]
print("\nPrediction:", model_gap.predict(test_text))

# ==========================================
# VISUALIZATION SECTION
# ==========================================

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# ------------------------------
# 1. CONFUSION MATRIX
# ------------------------------

cm = confusion_matrix(y_test, preds, labels=["understood", "not_understood"])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["understood", "not_understood"],
            yticklabels=["understood", "not_understood"])

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Concept Gap Model")
plt.tight_layout()
plt.show()


# ------------------------------
# 2. CLASS DISTRIBUTION GRAPH
# ------------------------------

plt.figure(figsize=(6,4))
df["label"].value_counts().plot(kind="bar")
plt.title("Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# ------------------------------
# 3. PRECISION / RECALL / F1 GRAPH
# ------------------------------

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, preds, average="weighted"
)

metrics = ["Precision", "Recall", "F1 Score"]
values = [precision, recall, f1]

plt.figure(figsize=(6,4))
plt.bar(metrics, values)
plt.title("Model Performance Metrics")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()


# ------------------------------
# 4. PREDICTION CONFIDENCE (OPTIONAL)
# ------------------------------

probs = model_gap.predict_proba(X_test)

confidence = probs.max(axis=1)

plt.figure(figsize=(6,4))
plt.hist(confidence, bins=20)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

