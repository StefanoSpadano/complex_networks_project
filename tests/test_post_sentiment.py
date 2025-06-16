# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 18:53:45 2025

@author: Raffaele
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from analize_sentiment import calculate_sentiment, add_sentiment_to_posts, clean_sentiment_data
import pandas as pd


def test_calculate_sentiment_valid_string():
    """
    Given a string of text taken from a post or comment of a post coming from the subreddit taken into account,
    when the calculate_sentiment fuction is called,
    then the output should be a value rangng between -1 and 1.
    """
    #Initialize sentiment analyzer from Vader library
    analyzer = SentimentIntensityAnalyzer()
    #Initialize a string of text
    text = "One Piece is the best anime!"
    #write into the score variable the result of the calculate_sentiment function
    score = calculate_sentiment(text, analyzer)
    #asserts
    assert -1 <= score <= 1, "Sentiment score should be within [-1, 1]"

def test_calculate_sentiment_empty_string():
    """
    Given an empty string,
    when the calculate_sentiment function is called,
    then the result should be 0 or between -0.1 and 0.1 (ranges I've selected from the analize_Sentiment.py script).
    """
    #Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    #Initialize an empty string
    empty_string = ""
    #call the function calculate_sentiment passing the analyzer and the empty string
    score = calculate_sentiment(empty_string, analyzer)
    #asserts
    assert score == 0.0 or abs(score) < 0.1
 
def test_calculate_sentiment_non_string_input():
    """
    Given a non string input,
    when the calculate_sentiment is called passing the non string input,
    then the result should be 0.0
    """
    #Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    #Initialize 2 examples of non string inputs
    not_a_string_1 = 123
    not_a_string_2 = None
    #call the function calculate_sentiment passing the two non string inputs and the analyzer
    score_1 = calculate_sentiment(not_a_string_1, analyzer)
    score_2 = calculate_sentiment(not_a_string_2, analyzer)
    #asserts
    assert score_1 == 0.0
    assert score_2 == 0.0
    
def test_add_sentiment_to_posts_basic():
    """
    Given a dataframe containing selftext of mixed sentiments,
    when the function add_sentiment_to_posts is called,
    then the sentiments are correctly added in another column of the dataframe.
    """
    #Initialize a dataframe containg 4 different selftext which we can categorize in different sentiments
    df = pd.DataFrame({
        "selftext": ["This is great!", "I hate this", "Meh.", None]
    })
    #Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    #Initialize result calling the add_sentiment_to_posts function and passing a copy
    #of the dataframe just defined alongside the analyzer
    result = add_sentiment_to_posts(df.copy(), analyzer)
    #asserts
    #there is a column for the sentiment_selftext
    assert "sentiment_selftext" in result.columns
    #the result variable has the correct dimensions corresponsing to the number of selftext initialized
    assert len(result) == 4
    #at least 3 are strings entry
    assert result["sentiment_selftext"].notnull().sum() >= 3


def test_add_sentiment_to_posts_with_empty_df():
    """
    Given an empty dataframe which only contains a column for the selftexts but no actual selftexts,
    when the function add_sentiments_to_posts is called,
    then an empty dataframe should be returned.
    """
    #Initialize an empty dataframe with a column called selftext
    df = pd.DataFrame(columns=["selftext"])
    #Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    #Initialize the result variable
    result = add_sentiment_to_posts(df, analyzer)
    #assert
    assert result.empty

def test_add_sentiments_to_posts_with_NaN_selftexts():
    """
    Given a dataframe containg only NaN values under the "selftext" column,
    when the add_sentiment_to_posts function is called,
    then the sum of all sentiment scores should be zero.
    """
    #Initialize a dataframe which contains only NaN values under the column "selftext"
    df = pd.DataFrame({"selftext":[None, None, None]
    })
    #Initialize the sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    #Initialize the result variable
    result = add_sentiment_to_posts(df, analyzer)
    #assert
    assert result["sentiment_selftext"].sum() == 0.0


def test_clean_sentiment_data_with_nans():
    """
    Given a dataframe containing selftexts and one of them is a NaN value,
    when the clean_sentiment_data function is called,
    then the resulting dataframe should not include the corresponding line for that missing value.
    """
    #Initialize a dataframe with a selftext column containing a NaN value,
    #plus a column containing the corresponding sentiment scores
    df = pd.DataFrame({
        "selftext": ["Text 1", None, "Text 2"],
        "sentiment_selftext": [0.1, 0.0, -0.2]
    })
    #Call the function to clean the data
    cleaned = clean_sentiment_data(df)
    #asserts
    assert len(cleaned) == 2
    assert cleaned.isnull().sum().sum() == 0

def test_clean_sentiment_data_all_nans():
    """
    Given a dataframe containing all nans in the selftext column,
    when the clean_sentiment_data function is called,
    then should return an empty dataframe.
    """
    #Initialize a dataframe with all nan values in the selftext column
    df = pd.DataFrame({"selftext":[None, None, None]})
    #Call the function clean_sentiment_data
    cleaned = clean_sentiment_data(df)
    #assert
    assert cleaned.empty