# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 11:55:48 2025

@author: Raffaele
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from analize_comment_sentiment import (
    filter_comments, calculate_sentiment, add_sentiment_to_comments,
    add_sentiment_category, save_filtered_comments
)
from utils import categorize_sentiment
import pandas as pd
import tempfile
import os


def test_filter_comments_with_deleted_authors():
    """
    Given a dataframe containing comments with some deleted authors,
    when the filter_comments function is called,
    then the resulting dataframe should not contain any deleted authors.
    """
    #Initialize a dataframe with some deleted authors
    df = pd.DataFrame({
        "author": ["user1", "[deleted]", "user2", None],
        "body": ["Great comment!", "This was deleted", "Another comment", "Valid comment"],
        "score": [5, 10, 3, 7]
    })
    #Call the filter_comments function
    filtered = filter_comments(df)
    #asserts
    assert "[deleted]" not in filtered["author"].values
    assert not filtered["author"].isna().any()


def test_filter_comments_with_empty_bodies():
    """
    Given a dataframe containing comments with empty or whitespace-only bodies,
    when the filter_comments function is called,
    then the resulting dataframe should not contain any empty bodies.
    """
    #Initialize a dataframe with empty and whitespace bodies
    df = pd.DataFrame({
        "author": ["user1", "user2", "user3", "user4"],
        "body": ["Valid comment", "", "   ", "Another valid comment"],
        "score": [5, 10, 3, 7]
    })
    #Call the filter_comments function
    filtered = filter_comments(df)
    #asserts
    assert len(filtered) == 2  # Only valid comments should remain
    assert not (filtered["body"].str.strip() == "").any()


def test_filter_comments_with_low_scores():
    """
    Given a dataframe containing comments with scores less than or equal to 0,
    when the filter_comments function is called,
    then the resulting dataframe should only contain comments with positive scores.
    """
    #Initialize a dataframe with various scores including negative and zero
    df = pd.DataFrame({
        "author": ["user1", "user2", "user3", "user4"],
        "body": ["Comment 1", "Comment 2", "Comment 3", "Comment 4"],
        "score": [5, 0, -2, 3]
    })
    #Call the filter_comments function
    filtered = filter_comments(df)
    #asserts
    assert len(filtered) == 2  # Only positive scores should remain
    assert (filtered["score"] > 0).all()
    
def test_filter_comments_combined_filters():
    """
    Given a dataframe with multiple issues (deleted authors, empty bodies, low scores),
    when the filter_comments function is called,
    then only valid comments should remain.
    """
    #Initialize a dataframe with multiple issues
    df = pd.DataFrame({
        "author": ["user1", "[deleted]", "user2", None, "user3"],
        "body": ["Valid comment", "Deleted comment", "", "NaN author comment", "Another valid"],
        "score": [5, 10, 3, 7, -1]
    })
    #Call the filter_comments function
    filtered = filter_comments(df)
    #asserts
    assert len(filtered) == 1  # Only one comment should pass all filters
    assert filtered.iloc[0]["author"] == "user1"

def test_calculate_sentiment_positive_text():
    """
    Given a positive text string,
    when the calculate_sentiment function is called,
    then the result should be a positive value between 0 and 1.
    """
    #Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    # Initialize positive text
    text = "I absolutely love One Piece! It's amazing!"
    #Calculate sentiment
    score = calculate_sentiment(text, analyzer)
    #asserts
    assert score > 0, "Positive text should have positive sentiment"
    assert -1 <= score <= 1, "Sentiment score should be within [-1, 1]"

def test_calculate_sentiment_negative_text():
    """
    Given a negative text string,
    when the calculate_sentiment function is called,
    then the result should be a negative value between -1 and 0.
    """
    #Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    #Initialize negative text
    text = "I hate this show! It's terrible and boring!"
    #Calculate sentiment
    score = calculate_sentiment(text, analyzer)
    #asserts
    assert score < 0, "Negative text should have negative sentiment"
    assert -1 <= score <= 1, "Sentiment score should be within [-1, 1]"

def test_calculate_sentiment_non_string_input():
    """
    Given a non-string input,
    when the calculate_sentiment function is called,
    then the result should be 0.0.
    """
    #Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    # Initialize non-string inputs
    non_string_1 = 123
    non_string_2 = None
    non_string_3 = []
    #Calculate sentiment for each
    score_1 = calculate_sentiment(non_string_1, analyzer)
    score_2 = calculate_sentiment(non_string_2, analyzer)
    score_3 = calculate_sentiment(non_string_3, analyzer)
    #asserts
    assert score_1 == 0.0
    assert score_2 == 0.0
    assert score_3 == 0.0
    
def test_add_sentiment_to_comments_basic():
    """
    Given a dataframe containing comment bodies of mixed sentiments,
    when the add_sentiment_to_comments function is called,
    then the sentiments are correctly added in a new column of the dataframe.
    """
    #Initialize a dataframe with different sentiment comments
    df = pd.DataFrame({
        "author": ["user1", "user2", "user3", "user4"],
        "body": ["This is great!", "I hate this", "It's okay", "Amazing content!"],
        "score": [5, 3, 2, 8]
    })
    #Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    #Add sentiment to comments
    result = add_sentiment_to_comments(df.copy(), analyzer)
    #asserts
    assert "sentiment_body" in result.columns
    assert len(result) == 4
    assert result["sentiment_body"].notnull().sum() == 4

def test_add_sentiment_to_comments_with_empty_df():
    """
    Given an empty dataframe with only column headers,
    when the add_sentiment_to_comments function is called,
    then an empty dataframe with the sentiment column should be returned.
    """
    #Initialize empty dataframe with required columns
    df = pd.DataFrame(columns=["author", "body", "score"])
    #Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    #Add sentiment to comments
    result = add_sentiment_to_comments(df, analyzer)
    #asserts
    assert result.empty
    assert "sentiment_body" in result.columns
    
def test_add_sentiment_to_comments_with_mixed_body_types():
    """
    Given a dataframe containing both valid strings and NaN values in body column,
    when the add_sentiment_to_comments function is called,
    then valid strings should get sentiment scores and NaN should get 0.0.
    """
    #Initialize dataframe with mixed body types
    df = pd.DataFrame({
        "author": ["user1", "user2", "user3"],
        "body": ["Valid negative comment", None, "This comment is really positive"],
        "score": [5, 3, 7]
    })
    #Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    #Add sentiment to comments
    result = add_sentiment_to_comments(df, analyzer)
    #asserts
    assert "sentiment_body" in result.columns
    assert result.iloc[0]["sentiment_body"] != 0.0  
    assert result.iloc[1]["sentiment_body"] == 0.0  
    assert result.iloc[2]["sentiment_body"] != 0.0
    
def test_add_sentiment_category_basic():
    """
    Given a dataframe with sentiment_body scores,
    when the add_sentiment_category function is called,
    then a sentiment_category column should be added with appropriate categories.
    """
    #Initialize dataframe with sentiment scores
    df = pd.DataFrame({
        "author": ["user1", "user2", "user3"],
        "body": ["Great!", "Terrible", "Okay"],
        "score": [5, 3, 4],
        "sentiment_body": [0.6, -0.5, 0.0]
    })
    #Apply the categorize_sentiment function from utils
    result_1 = categorize_sentiment(df.iloc[0]["sentiment_body"])
    result_2 = categorize_sentiment(df.iloc[1]["sentiment_body"])
    result_3 = categorize_sentiment(df.iloc[2]["sentiment_body"])
    assert result_1 == "Positive"
    assert result_2 == "Negative"
    assert result_3 == "Neutral"
    
def test_add_sentiment_category_edges():
    """
    Given a dataframe containing sentiment_body scores in proximity of the edges,
    when the add_sentiment_category function is called,
    then the correct sentiment category should be returned.
    """
    #Initialize a dataframe with sentiment scores
    df = pd.DataFrame({
        "author":["user1", "user2", "user3"],
        "sentiment_body":[0.1000000000000001, -0.1000000000000001, 0.0999999999999999]
        })
    #Apply the sentiment category function from utils
    result_1 = categorize_sentiment(df.iloc[0]["sentiment_body"])
    result_2 = categorize_sentiment(df.iloc[1]["sentiment_body"])
    result_3 = categorize_sentiment(df.iloc[2]["sentiment_body"])
    #asserts
    assert result_1 == "Positive"
    assert result_2 == "Negative"
    assert result_3 == "Neutral"

def test_save_filtered_comments_basic(tmp_path):
    """
    Given a dataframe with filtered comments,
    when the save_filtered_comments function is called,
    then it should save the data to the specified CSV file path.
    """
    #Initialize example dataframe
    df = pd.DataFrame({
        "author": ["user1", "user2"],  
        "body": ["Comment 1", "Comment 2"],
        "score": [5, 8],
        "sentiment_body": [0.2, 0.7]
    })
    #Create temporary file path
    path = tmp_path / "test_comments.csv"
    #Save filtered comments
    save_filtered_comments(df, str(path))
    #Load and verify saved data
    loaded = pd.read_csv(path)
    #asserts
    assert len(loaded) == len(df)
    assert list(loaded.columns) == list(df.columns)
    assert loaded["author"].tolist() == df["author"].tolist()

def test_save_filtered_comments_empty_df(tmp_path):
    """
    Given an empty dataframe,
    when the save_filtered_comments function is called,
    then it should save an empty CSV file with headers.
    """
    #Initialize empty dataframe with columns
    df = pd.DataFrame(columns=["author", "body", "score", "sentiment_body"])
    #Create temporary file path
    path = tmp_path / "empty_comments.csv"
    #Save filtered comments
    save_filtered_comments(df, str(path))
    #Load and verify saved data
    loaded = pd.read_csv(path)
    #asserts
    assert loaded.empty
    assert list(loaded.columns) == list(df.columns)
    
def test_save_filtered_comments_file_creation(tmp_path):
    """
    Given a valid dataframe and a file path,
    when the save_filtered_comments function is called,
    then the CSV file should be created at the specified location.
    """
    #Initialize dataframe
    df = pd.DataFrame({
        "author": ["user1"],
        "body": ["Test comment"],
        "score": [10],
        "sentiment_body": [0.5]
    })
    #Create file path
    path = tmp_path / "created_file.csv"  
    #Save filtered comments
    save_filtered_comments(df, str(path))
    #asserts
    assert path.exists(), "CSV file should be created"
    assert path.is_file(), "Path should point to a file"