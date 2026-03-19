# =========================================================
# BERT — MULTIWOZ DIALOGUE CLASSIFICATION + VISUALS
# Uses your exact dataset path
# =========================================================

import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm


# =========================================================
# LOAD YOUR MULTIWOZ DATA
# =========================================================

file_path = r"C:\Users\mansi\Downloads\SNLP MODEL\archive (2)\MultiWOZ_2.2\train\dialogues_017.json"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Dialogues loaded:", len(data))


# =========================================================
# PREPARE TEXT + LABELS
# =========================================================

texts = []
labels = []

for dialog in data[:500]:   # small subset for speed
    turns = dialog["turns"]

    # Combine conversation into single text
    conversation = " ".join([t["utterance"] for t in turns])
    texts.append(conversation)

    # Label complexity
    if len(turns) < 5:
        labels.append("low")
    elif len(turns) < 10:
        labels.append("medium")
    else:
        labels.append("high")


# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42
)


# =========================================================
# TOKENIZER
# =========================================================

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# =========================================================
# DATASET CLASS
# =========================================================

class MultiWOZDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=128
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = MultiWOZDataset(train_texts, train_labels)
test_dataset = MultiWOZDataset(test_texts, test_labels)


# =========================================================
# LOAD BERT MODEL
# =========================================================

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# =========================================================
# TRAINING
# =========================================================

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
optimizer = AdamW(model.parameters(), lr=2e-5)

losses = []

model.train()
print("\nTraining BERT...\n")

for epoch in range(2):   # keep small for speed
    loop = tqdm(train_loader, leave=True)

    for batch in loop:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)

        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels_batch
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())


# =========================================================
# EVALUATION
# =========================================================

model.eval()

correct = 0
total = 0
y_true = []
y_pred = []

with torch.no_grad():
    for item in test_dataset:
        input_ids = item["input_ids"].unsqueeze(0).to(device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(device)
        label = item["labels"].item()

        outputs = model(input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).item()

        y_true.append(label)
        y_pred.append(pred)

        if pred == label:
            correct += 1
        total += 1

accuracy = correct / total
print("\nBERT Accuracy:", round(accuracy, 4))


# =========================================================
# SAMPLE PREDICTION
# =========================================================

sample_text = test_texts[0]

inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=128
).to(device)

outputs = model(**inputs)
pred_class = torch.argmax(outputs.logits, dim=1).item()

print("Predicted Dialogue Complexity:",
      le.inverse_transform([pred_class])[0])


# =========================================================
# VISUALIZATION
# =========================================================

# ---- Loss Curve ----
plt.figure()
plt.plot(losses)
plt.title("BERT Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.show()


# ---- Accuracy Bar ----
plt.figure()
plt.bar(["Accuracy"], [accuracy])
plt.ylim(0, 1)
plt.title("BERT Accuracy")
plt.show()


# ---- Confusion Matrix ----
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=le.classes_
)

disp.plot()
plt.title("BERT Confusion Matrix")
plt.show()
