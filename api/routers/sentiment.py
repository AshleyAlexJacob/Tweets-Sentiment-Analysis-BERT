from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from api.schemas.sentiment import SentimentRequest, SentimentResponse
from api.dependencies import get_model, get_preprocessor

router = APIRouter()

@router.post(
    "/predict",
    response_model=SentimentResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict tweet sentiment",
    description="Analyzes the sentiment of a single tweet using the BERT model."
)
def predict_sentiment(
    request: SentimentRequest,
    model: Any = Depends(get_model),
    preprocessor: Any = Depends(get_preprocessor)
) -> SentimentResponse:
    """Predicts sentiment for an input tweet.

    Args:
        request (SentimentRequest): The input text.
        model (Any): The loaded BERT model (injected).
        preprocessor (Any): The initialized preprocessor (injected).

    Returns:
        SentimentResponse: The prediction results.
    """
    # Business logic would call src/model/prediction.py here
    # For now, returning a mock response
    return SentimentResponse(
        text=request.text,
        sentiment="positive",
        confidence=0.95,
        class_id=1
    )
