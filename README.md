# Intelligent Tutoring System

An AI-powered tutoring system that learns from online course content 
and lecture notes — capable of answering subject-specific student 
doubts, generating concise short notes, and quizzing students on 
any topic from the learned material.

## What it does
- Answers subject-specific doubts based on course content
- Generates concise short notes on any topic
- Quizzes students to test their understanding

## Models Used
Experimented with and compared multiple NLP architectures:
- LSTM
- GRU
- LSTM + GRU with GloVe embeddings
- Fine-tuned BERT (best performance)

## Tech Stack
- Python
- PyTorch
- BERT (fine-tuned via Hugging Face Transformers)
- LSTM, GRU
- GloVe Embeddings
- Hugging Face Transformers

## Dataset
Trained on online course content and lecture notes.  
Large dataset files are not included in this repo due to size.  
Access datasets via Hugging Face Hub.

## How to Run
1. Clone the repo
   git clone https://github.com/mansidugar/Intelligent-Tutor-System.git

2. Install dependencies
   pip install -r requirements.txt

3. Run the model
   python fine_tune_its.py
