import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from analize_sentiment import categorize_sentiment


def test_categorize_sentiment():
    # Test negative score
    assert categorize_sentiment(-0.5) == 'Negative'
    
    # Test borderline negative
    assert categorize_sentiment(-0.11) == 'Negative'
    
    # Test neutral score
    assert categorize_sentiment(0.0) == 'Neutral'
    
    # Test borderline positive
    assert categorize_sentiment(0.1) == 'Neutral'
    
    # Test positive score
    assert categorize_sentiment(0.5) == 'Positive'
