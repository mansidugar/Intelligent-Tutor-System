import os
import re
import sys
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION  —  Edit these paths / settings before running
# ══════════════════════════════════════════════════════════════════════════════

# Path to the Excel dataset created in the previous step
DATA_PATH = r"C:\Users\mansi\Downloads\SNLP MODEL\natural_processes_dataset.xlsx"

# Where the fine-tuned model will be saved
OUTPUT_DIR = "./its_finetuned_model"

# Base model — distilgpt2 is fast & lightweight (works on CPU too)
# For better quality: "gpt2"  |  "gpt2-medium"  |  "microsoft/phi-2"
MODEL_NAME = "distilgpt2"

MAX_LENGTH   = 512
BATCH_SIZE   = 4
EPOCHS       = 3
LEARNING_RATE = 2e-5
USE_LORA     = True   # True = only trains ~1% of params, much faster

# Words/phrases that signal the student wants to end the chat
EXIT_PHRASES = [
    "thank you", "thanks", "bye", "goodbye", "see you",
    "that's all", "thats all", "i'm done", "im done",
    "no more questions", "exit", "quit", "stop", "end",
    "i got it", "i understand now", "got it, thanks",
]

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — Load the Excel Dataset
# ══════════════════════════════════════════════════════════════════════════════

def load_excel_dataset(path: str) -> pd.DataFrame:
    """
    Reads the natural_processes_dataset.xlsx file.
    Uses the 'Natural Processes Dataset' sheet which has:
      - Process Name, Definition, Key Equation, Steps, Key Components,
        Ecological Significance, Sample Tutoring Questions, Related Processes
    """
    print(f"\n[1/5] Loading dataset from:\n      {path}\n")

    if not Path(path).exists():
        print(f"  ❌  ERROR: File not found at:\n      {path}")
        print("  Please update DATA_PATH at the top of this script.\n")
        sys.exit(1)

    df = pd.read_excel(path, sheet_name="Natural Processes Dataset", header=1)
    df.columns = df.columns.str.strip()
    df.dropna(subset=["Process Name", "Definition"], inplace=True)
    print(f"  ✅  Loaded {len(df)} natural processes from Excel.\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — Build a Knowledge Base (for rule-based fallback in chat)
# ══════════════════════════════════════════════════════════════════════════════

def build_knowledge_base(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Converts each Excel row into a searchable knowledge entry.
    Used by the chat engine to find the best matching answer.
    """
    kb = {}
    for _, row in df.iterrows():
        name = str(row.get("Process Name", "")).strip().lower()
        kb[name] = {
            "name":       str(row.get("Process Name", "")).strip(),
            "definition": str(row.get("Definition", "")).strip(),
            "equation":   str(row.get("Key Equation / Reaction", "")).strip(),
            "steps":      str(row.get("Steps / Pathway", "")).strip(),
            "components": str(row.get("Key Components", "")).strip(),
            "significance": str(row.get("Ecological Significance", "")).strip(),
            "difficulty": str(row.get("Difficulty", "")).strip(),
            "questions":  str(row.get("Sample Tutoring Questions", "")).strip(),
            "related":    str(row.get("Related Processes", "")).strip(),
        }
    return kb


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — Convert Dataset Rows into Conversation Training Strings
# ══════════════════════════════════════════════════════════════════════════════

def build_training_conversations(df: pd.DataFrame) -> List[str]:
    """
    Converts each row into multiple Student ↔ Tutor exchanges.
    Each exchange is a self-contained training example.

    Format:
        <|ITS|> System context
        Student: question
        Tutor: detailed answer
        <|endoftext|>
    """
    print("[2/5] Building training conversations from dataset...\n")
    conversations = []

    for _, row in df.iterrows():
        name = str(row.get("Process Name", "")).strip()
        defn = str(row.get("Definition", "")).strip()
        eq   = str(row.get("Key Equation / Reaction", "")).strip()
        steps = str(row.get("Steps / Pathway", "")).strip()
        comps = str(row.get("Key Components", "")).strip()
        sig  = str(row.get("Ecological Significance", "")).strip()
        diff = str(row.get("Difficulty", "")).strip()
        raw_qs = str(row.get("Sample Tutoring Questions", "")).strip()
        related = str(row.get("Related Processes", "")).strip()

        system_ctx = (
            "<|ITS|> You are a friendly and knowledgeable science tutor. "
            "Explain concepts clearly and encourage the student."
        )

        # — Conversation 1: What is it?
        conversations.append(
            f"{system_ctx}\n"
            f"Student: What is {name}?\n"
            f"Tutor: Great question! {name} is {defn} "
            f"The key components involved are: {comps}.\n"
            f"<|endoftext|>"
        )

        # — Conversation 2: How does it work? (steps)
        if steps and steps != "nan":
            conversations.append(
                f"{system_ctx}\n"
                f"Student: How does {name} work step by step?\n"
                f"Tutor: Sure! Here is how {name} works: {steps} "
                f"This process has a difficulty level of {diff}, so take your time understanding it!\n"
                f"<|endoftext|>"
            )

        # — Conversation 3: Equation / reaction
        if eq and eq != "nan":
            conversations.append(
                f"{system_ctx}\n"
                f"Student: What is the equation or reaction for {name}?\n"
                f"Tutor: The key equation for {name} is: {eq} "
                f"This summarizes what goes in and what comes out of the process.\n"
                f"<|endoftext|>"
            )

        # — Conversation 4: Why does it matter?
        if sig and sig != "nan":
            conversations.append(
                f"{system_ctx}\n"
                f"Student: Why is {name} important?\n"
                f"Tutor: {name} is very important because: {sig} "
                f"It plays a vital role in the broader ecosystem.\n"
                f"<|endoftext|>"
            )

        # — Conversation 5: Related processes
        if related and related != "nan":
            conversations.append(
                f"{system_ctx}\n"
                f"Student: What topics are related to {name}?\n"
                f"Tutor: {name} is closely connected to: {related}. "
                f"Understanding these related processes will give you a much deeper picture!\n"
                f"<|endoftext|>"
            )

        # — Conversation 6: Sample Q&A from dataset
        if raw_qs and raw_qs != "nan":
            questions = [q.strip() for q in raw_qs.split("|") if q.strip()]
            for q in questions:
                conversations.append(
                    f"{system_ctx}\n"
                    f"Student: {q}\n"
                    f"Tutor: That is a great question about {name}! "
                    f"{defn} To be more specific — {steps} "
                    f"The main components are {comps}, and the significance is: {sig}\n"
                    f"<|endoftext|>"
                )

        # — Conversation 7: Student says goodbye mid-topic
        conversations.append(
            f"{system_ctx}\n"
            f"Student: Thank you, I think I understand {name} now!\n"
            f"Tutor: You're very welcome! You did a great job learning about {name}. "
            f"Feel free to come back anytime if you have more questions. Good luck!\n"
            f"<|endoftext|>"
        )

    print(f"  ✅  Created {len(conversations)} training conversation examples.\n")
    return conversations


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — PyTorch Dataset Class
# ══════════════════════════════════════════════════════════════════════════════

class ITSDataset(Dataset):
    """Tokenizes all training conversations for the Trainer."""
    def __init__(self, texts: List[str], tokenizer, max_length: int):
        print(f"[2.5/5] Tokenizing {len(texts)} examples (this may take a moment)...\n")
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels":         self.encodings["input_ids"][idx].clone(),
            # labels = input_ids → model learns to predict next token (causal LM)
        }


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5 — Load Model + Apply LoRA
# ══════════════════════════════════════════════════════════════════════════════

def load_model_and_tokenizer(model_name: str):
    """
    Loads the base language model and tokenizer.
    If USE_LORA=True, wraps the model with LoRA adapters so only
    ~1% of parameters are trainable — making training fast even on CPU.
    """
    print(f"[3/5] Loading base model: '{model_name}'\n")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT-2 family has no padding token by default — use EOS token instead
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)

    if USE_LORA:
        print("      Applying LoRA (Parameter-Efficient Fine-Tuning)...")
        print("      This trains only a tiny fraction of weights — fast & memory-efficient.\n")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,              # LoRA rank — higher = more capacity, more memory
            lora_alpha=32,    # Scaling factor
            lora_dropout=0.1, # Regularization
            target_modules=["c_attn"],  # GPT-2 attention projection layers
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print()

    return model, tokenizer


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6 — Training
# ══════════════════════════════════════════════════════════════════════════════

def train_model(model, tokenizer, dataset: ITSDataset):
    """Runs the HuggingFace Trainer loop."""
    print("[4/5] Starting fine-tuning...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"      Device: {device}  {'(GPU — fast!)' if device == 'cuda' else '(CPU — slower, be patient)'}\n")
    model.to(device)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        save_steps=500,
        save_total_limit=2,          # Keep only last 2 checkpoints to save disk
        logging_steps=50,
        prediction_loss_only=True,
        fp16=torch.cuda.is_available(),  # Use half-precision on GPU for speed
        report_to="none",                # Disables W&B / TensorBoard logging
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,   # Causal LM (predict next token), NOT masked LM like BERT
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    print("\n  ✅  Training complete!\n")
    return trainer


def save_model(model, tokenizer):
    print(f"[5/5] Saving fine-tuned model to: {OUTPUT_DIR}\n")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("  ✅  Model saved successfully!\n")


# ══════════════════════════════════════════════════════════════════════════════
#  CHAT ENGINE — Rule-Based Knowledge Lookup (fast & reliable fallback)
# ══════════════════════════════════════════════════════════════════════════════

def find_best_match(question: str, kb: Dict[str, Dict]) -> Dict | None:
    """
    Searches the knowledge base for a process matching the student's question.
    Uses simple keyword matching — works great for subject-specific tutoring.
    """
    question_lower = question.lower()

    # Exact match first
    for key, entry in kb.items():
        if key in question_lower:
            return entry

    # Partial word match
    for key, entry in kb.items():
        words = key.split()
        if any(word in question_lower for word in words if len(word) > 3):
            return entry

    return None


def build_tutor_response(question: str, kb: Dict[str, Dict]) -> str:
    """
    Generates a tutor response for the student's question.
    Tries to find a matching topic in the knowledge base.
    Falls back gracefully if nothing is found.
    """
    match = find_best_match(question, kb)

    if match:
        name = match["name"]
        defn = match["definition"]
        eq   = match["equation"]
        steps = match["steps"]
        sig  = match["significance"]
        related = match["related"]

        # Build a natural tutoring response
        response_parts = [f"Great question! Let me explain **{name}**.\n"]
        response_parts.append(f"{defn}")

        if eq and eq.lower() != "nan":
            response_parts.append(f"\n\n📐 Key equation/reaction:\n  {eq}")

        if steps and steps.lower() != "nan":
            response_parts.append(f"\n\n🔄 How it works step by step:\n  {steps}")

        if sig and sig.lower() != "nan":
            response_parts.append(f"\n\n🌍 Why it matters:\n  {sig}")

        if related and related.lower() != "nan":
            response_parts.append(f"\n\n🔗 Related topics you might want to explore:\n  {related}")

        response_parts.append(
            "\n\nDoes that help? Feel free to ask me anything else about this "
            "or any other natural process! 😊"
        )
        return "".join(response_parts)

    else:
        # Graceful fallback — tutor doesn't pretend to know everything
        return (
            "Hmm, that's a great question! I don't have detailed information "
            "on that specific topic in my current knowledge base. "
            "Here's what I can tell you — it's likely related to one of the "
            "natural processes I know, such as photosynthesis, the water cycle, "
            "the nitrogen cycle, or geological processes.\n\n"
            "Could you rephrase your question or ask about a specific process? "
            "I'm here to help with whatever I can! 😊"
        )


def is_exit_message(message: str) -> bool:
    """Returns True if the student's message signals they want to end the chat."""
    msg = message.lower().strip()
    return any(phrase in msg for phrase in EXIT_PHRASES)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE CHAT LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_chat(kb: Dict[str, Dict], model=None, tokenizer=None):
    """
    Starts the interactive tutoring session.

    Conversation flow:
      - Student types a question
      - Tutor searches the knowledge base and gives a detailed answer
      - If unknown → tutor gives a nearby/helpful response
      - If student says 'thank you', 'bye', etc. → session ends gracefully
      - Loop continues until exit phrase detected
    """

    print("\n" + "═" * 65)
    print("  🌿  INTELLIGENT TUTORING SYSTEM — Natural Processes")
    print("  📚  Subject: Biology, Earth Science, Ecology & More")
    print("═" * 65)
    print("\n  Hi! I'm your science tutor. Ask me anything about natural")
    print("  processes — photosynthesis, the water cycle, plate tectonics,")
    print("  the nitrogen cycle, and much more!")
    print("\n  Type 'bye', 'thank you', or 'exit' whenever you're done.\n")
    print("─" * 65 + "\n")

    turn = 1
    while True:
        # ── Get student input ──────────────────────────────────────────────
        try:
            student_input = input(f"  You (Turn {turn}): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Tutor: It was great studying with you! Goodbye! 👋\n")
            break

        if not student_input:
            print("  Tutor: Feel free to ask me anything! I'm listening. 😊\n")
            continue

        # ── Check for exit ─────────────────────────────────────────────────
        if is_exit_message(student_input):
            print(
                "\n  Tutor: You're very welcome! I'm so glad I could help you "
                "learn today.\n"
                "         Keep exploring the amazing world of natural processes!\n"
                "         Goodbye, and good luck with your studies! 🌱👋\n"
            )
            print("─" * 65)
            break

        # ── Generate tutor response ────────────────────────────────────────
        print("\n  Tutor: ", end="", flush=True)
        response = build_tutor_response(student_input, kb)

        # Print response line by line for readability
        for line in response.split("\n"):
            print(f"  {line}" if not line.startswith("  ") else line)

        print()  # blank line between turns
        turn += 1


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN — Orchestrates training then launches chat
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "═" * 65)
    print("  ITS Pipeline: Natural Processes Tutoring System")
    print("═" * 65)

    # ── Load Excel Dataset ─────────────────────────────────────────────────
    df = load_excel_dataset(DATA_PATH)

    # ── Build knowledge base (used in chat) ───────────────────────────────
    knowledge_base = build_knowledge_base(df)

    # ── Decide: Train or Chat? ─────────────────────────────────────────────
    already_trained = Path(OUTPUT_DIR).exists() and any(Path(OUTPUT_DIR).iterdir())

    if already_trained:
        print(f"\n  ℹ️  Found existing fine-tuned model at: {OUTPUT_DIR}")
        choice = input("  Re-train from scratch? (y/n, default=n): ").strip().lower()
        skip_training = choice != "y"
    else:
        skip_training = False

    model, tokenizer = None, None

    if not skip_training:
        # Build training data from Excel
        conversations = build_training_conversations(df)

        # Load base model
        model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

        # Tokenize
        dataset = ITSDataset(conversations, tokenizer, MAX_LENGTH)
        print(f"  ✅  Dataset: {len(dataset)} training samples\n")

        # Train
        trainer = train_model(model, tokenizer, dataset)

        # Save
        save_model(model, tokenizer)

    # ── Launch Interactive Chat ────────────────────────────────────────────
    run_chat(knowledge_base, model, tokenizer)
