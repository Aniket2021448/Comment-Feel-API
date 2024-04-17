from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

class BatchInput(BaseModel):
    comments: List[str]

# Load the BERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

def analyze_sentiments_batch(comments: List[str]):
    # Tokenize all comments at once
    tokenized_comments = tokenizer(comments, padding=True, truncation=True, return_tensors="pt")

    # Perform sentiment analysis in batch
    with torch.no_grad():
        outputs = model(**tokenized_comments)
        logits = outputs.logits

    # Determine sentiments for each comment
    sentiments = torch.argmax(logits, dim=1).tolist()
    
    return sentiments

@app.post("/analyze_sentiment_batch/")
def analyze_sentiment_batch(data: BatchInput):
    comments = data.comments

    if not comments:
        raise HTTPException(status_code=400, detail="No comments provided")

    sentiments = analyze_sentiments_batch(comments)

    # Count sentiments
    sentiment_counts = {
        "positive": sentiments.count(2),
        "negative": sentiments.count(0),
        "neutral": sentiments.count(1)
    }

    return sentiment_counts
