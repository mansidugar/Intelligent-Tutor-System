# ==========================================
# BAYESIAN CONFIDENCE MODEL (SQuAD)
# ==========================================

# pip install pandas numpy matplotlib seaborn scikit-learn openpyxl

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# LOAD YOUR DATASET
# ==========================================

file_path = r"C:\Users\mansi\Downloads\SNLP MODEL\squad_2.0_train_complete_130319.xlsx"

df = pd.read_excel(file_path)

print("Dataset loaded:", df.shape)
print("Columns:", df.columns)

# ==========================================
# FIND CONFIDENCE SIGNAL
# ==========================================

# Try to detect 'is_impossible' column automatically
possible_cols = [c.lower() for c in df.columns]

if "is_impossible" in possible_cols:
    col_name = df.columns[possible_cols.index("is_impossible")]
else:
    # fallback: assume empty answer = impossible
    col_name = None

# ==========================================
# CREATE CONFIDENCE LABEL
# ==========================================

confidence_scores = []

for i in range(min(1000, len(df))):   # use small subset
    if col_name is not None:
        impossible = df.iloc[i][col_name]
        conf = 0.2 if impossible == 1 else 0.9
    else:
        ans = str(df.iloc[i].to_dict())
        conf = 0.2 if ans.strip() == "" else 0.9
    
    confidence_scores.append(conf)

confidence_scores = np.array(confidence_scores)

print("\nSample Confidence Scores:")
print(confidence_scores[:20])

# ==========================================
# VISUAL 1 — CONFIDENCE DISTRIBUTION
# ==========================================

plt.figure(figsize=(6,4))
plt.hist(confidence_scores, bins=20)
plt.title("Confidence Score Distribution")
plt.xlabel("Confidence")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ==========================================
# VISUAL 2 — KNOWN vs UNKNOWN DETECTION
# ==========================================

known = np.sum(confidence_scores > 0.5)
unknown = np.sum(confidence_scores <= 0.5)

plt.figure(figsize=(5,5))
plt.pie([known, unknown],
        labels=["Answerable (High Confidence)", "Unanswerable (Low Confidence)"],
        autopct='%1.1f%%')
plt.title("Known vs Unknown Detection")
plt.show()

# ==========================================
# VISUAL 3 — CONFIDENCE OVER SAMPLES
# ==========================================

plt.figure(figsize=(10,4))
plt.plot(confidence_scores[:200])
plt.title("Confidence Score Across Samples")
plt.xlabel("Sample Index")
plt.ylabel("Confidence")
plt.tight_layout()
plt.show()

# ==========================================
# FUNCTION FOR YOUR TUTOR SYSTEM
# ==========================================

def bayesian_confidence(probabilities):
    return np.max(probabilities)

# Example usage (simulated probabilities)
example_probs = np.array([0.3, 0.7])
conf = bayesian_confidence(example_probs)

print("\nExample Confidence:", conf)

if conf < 0.6:
    print("Low confidence → retrieve verified content")
else:
    print("High confidence → safe to answer")
