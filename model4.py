# ==========================================
# HMM LEARNING STATE MODEL (MultiWOZ)
# ==========================================

# pip install hmmlearn pandas matplotlib seaborn numpy

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from collections import Counter

# ==========================================
# LOAD YOUR DATASET
# ==========================================

file_path = r"C:\Users\mansi\Downloads\SNLP MODEL\archive (2)\MultiWOZ_2.2\train\dialogues_001.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dialogues loaded:", len(data))

# ==========================================
# CREATE OBSERVATION SEQUENCE
# ==========================================

observations = []

for dialog in data[:200]:   # use first 200 dialogues
    turns = len(dialog["turns"])
    observations.append([turns])

observations = np.array(observations)

print("Observation shape:", observations.shape)

# ==========================================
# TRAIN HMM
# States = Beginner, Learning, Practicing, Mastered
# ==========================================

model_hmm = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=100)
model_hmm.fit(observations)

# ==========================================
# PREDICT STATES
# ==========================================

predicted_states = model_hmm.predict(observations)

print("\nPredicted States (first 20):")
print(predicted_states[:20])

# ==========================================
# VISUAL 1 — STATE DISTRIBUTION
# ==========================================

state_counts = Counter(predicted_states)

plt.figure(figsize=(6,4))
plt.bar(state_counts.keys(), state_counts.values())
plt.title("Learning State Distribution")
plt.xlabel("State (0=Beginner, 1=Learning, 2=Practicing, 3=Mastered)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ==========================================
# VISUAL 2 — STATE SEQUENCE OVER TIME
# ==========================================

plt.figure(figsize=(10,4))
plt.plot(predicted_states[:100], marker='o')
plt.title("Learning State Progression (First 100 Dialogues)")
plt.xlabel("Dialogue Index")
plt.ylabel("Predicted State")
plt.tight_layout()
plt.show()

# ==========================================
# VISUAL 3 — TRANSITION HEATMAP
# ==========================================

transitions = np.zeros((4,4))

for i in range(len(predicted_states)-1):
    transitions[predicted_states[i]][predicted_states[i+1]] += 1

plt.figure(figsize=(6,5))
sns.heatmap(transitions, annot=True, fmt=".0f", cmap="Blues")
plt.title("State Transition Heatmap")
plt.xlabel("Next State")
plt.ylabel("Current State")
plt.tight_layout()
plt.show()

# ==========================================
# TEST PREDICTION
# ==========================================

test_obs = np.array([[10]])   # dialogue with 10 turns
pred_state = model_hmm.predict(test_obs)

print("\nPredicted Learning Stage for test dialogue:", pred_state)
