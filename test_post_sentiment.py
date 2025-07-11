# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 18:53:45 2025

@author: Raffaele
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from analize_sentiment import calculate_sentiment

def test_calculate_sentiment_valid_string():
    analyzer = SentimentIntensityAnalyzer()
    text = "One Piece is the best anime!"
    score = calculate_sentiment(text, analyzer)
    assert -1 <= score <= 1, "Sentiment score should be within [-1, 1]"
