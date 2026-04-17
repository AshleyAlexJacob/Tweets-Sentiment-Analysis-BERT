from __future__ import annotations

from pydantic import BaseModel, Field


class SentimentRequest(BaseModel):
    """Request schema for sentiment analysis."""

    text: str = Field(
        ...,
        description="The tweet text to analyze",
        example="I love learning about AI!",
        min_length=1,
    )


class SentimentResponse(BaseModel):
    """Response schema for sentiment analysis."""

    text: str = Field(..., description="The original text analyzed")
    sentiment: str = Field(..., description="The predicted sentiment class")
    confidence: float = Field(..., description="Confidence score for the prediction")
    class_id: int = Field(..., description="Numerical class ID")
    probabilities: dict[str, float] = Field(
        ...,
        description="Softmax probability for negative, neutral, and positive",
    )
