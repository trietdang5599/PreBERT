import torch
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


_vader = SentimentIntensityAnalyzer()

def get_coarse_sentiment_score(model, tokenizer, text):
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    # Preserve uncertainty across all five classes. The former argmax mapping
    # saturated near 1.0 on imbalanced datasets and discarded useful mass from
    # minority ratings.
    rating_values = torch.arange(
        1,
        probabilities.shape[1] + 1,
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    expected_rating = torch.sum(probabilities[0] * rating_values)
    return ((expected_rating - 1.0) / max(probabilities.shape[1] - 1, 1)).item()


def get_vader_coarse_sentiment_score(text):
    """Return a deterministic [0, 1] score when no BERT head is trained."""
    compound = _vader.polarity_scores(text or "")["compound"]
    return (compound + 1.0) / 2.0
