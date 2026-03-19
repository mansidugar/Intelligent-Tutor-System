# ==========================================
# SBERT ANSWER SIMILARITY MODEL
# Dataset: ASAP Source Texts (Your File)
# ==========================================

# pip install sentence-transformers pandas matplotlib seaborn scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ==========================================
# LOAD YOUR DATASET
# ==========================================

file_path = r"C:\Users\mansi\Downloads\SNLP MODEL\ASAP2_train_sourcetexts.csv"

df = pd.read_csv(file_path)

print("Dataset loaded:", df.shape)
print(df.head())

# Assume column containing text is named "source_text"
# If different, print(df.columns) and change below

text_column = df.columns[0]   # usually first column contains text

texts = df[text_column].dropna().tolist()

# Use first 200 texts for faster demo
texts = texts[:200]

# ==========================================
# LOAD SBERT MODEL
# ==========================================

model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode texts
embeddings = model.encode(texts)

# ==========================================
# COMPUTE SIMILARITY BETWEEN RANDOM PAIRS
# ==========================================

similarities = []

for i in range(0, len(embeddings)-1, 2):
    sim = cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0]
    similarities.append(sim)

similarities = np.array(similarities)

print("\nSample Similarity Scores:")
print(similarities[:10])

# ==========================================
# VISUAL 1 — SIMILARITY DISTRIBUTION
# ==========================================

plt.figure(figsize=(6,4))
plt.hist(similarities, bins=20)
plt.title("Similarity Score Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# ==========================================
# VISUAL 2 — SCATTER PLOT
# ==========================================

plt.figure(figsize=(6,4))
plt.scatter(range(len(similarities)), similarities)
plt.title("Similarity Score vs Sample Index")
plt.xlabel("Sample Index")
plt.ylabel("Similarity")
plt.tight_layout()
plt.show()

# ==========================================
# VISUAL 3 — KDE PLOT
# ==========================================

sns.kdeplot(similarities, fill=True)
plt.title("Similarity Density Curve")
plt.xlabel("Similarity")
plt.tight_layout()
plt.show()
