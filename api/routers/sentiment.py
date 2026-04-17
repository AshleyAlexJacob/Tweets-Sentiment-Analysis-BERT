from __future__ import annotations

from typing import Annotated

import torch
from fastapi import APIRouter, Depends, status

from api.dependencies import get_model, get_preprocessor
from api.schemas.sentiment import SentimentRequest, SentimentResponse
from src.data.preprocessor import TweetPreprocessor
from src.model.architecture import BertSentimentClassifier

router = APIRouter()

_LABEL_NAMES: tuple[str, str, str] = ("negative", "neutral", "positive")


@router.post(
    "/predict",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict tweet sentiment",
    description="Analyzes the sentiment of a single tweet using the BERT model.",
)
def predict_sentiment(
    request: SentimentRequest,
    model: Annotated[BertSentimentClassifier, Depends(get_model)],
    preprocessor: Annotated[TweetPreprocessor, Depends(get_preprocessor)],
) -> SentimentResponse:
    """Predicts sentiment for an input tweet.

    Args:
        request: The input text.
        model: The loaded BERT model (injected).
        preprocessor: The initialized preprocessor (injected).

    Returns:
        The prediction results.
    """
    text = preprocessor.clean_text(request.text)
    encoding = preprocessor.tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=preprocessor.max_length,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    device = next(model.parameters()).device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.inference_mode():
        outputs = model(input_ids, attention_mask)
    logits = outputs["logits"]
    probs = torch.softmax(logits, dim=-1)[0]
    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())
    sentiment = _LABEL_NAMES[pred_id] if pred_id < len(_LABEL_NAMES) else str(pred_id)
    return SentimentResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=confidence,
        class_id=pred_id,
    )
