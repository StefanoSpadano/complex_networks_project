# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:57:50 2025

@author: Raffaele
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data_collection import RedditDataCollector  # adjust path as needed

#Temporarily replaces praw.Reddit class just for test function
@patch("praw.Reddit")
def test_fetch_posts_return_matching_flair(mock_reddit):
    
    """
    Given a mock subreddit with posts having different flairs,
    when the fetch function is called alongside a list of target flairs,
    then it should return only posts with that type of flair ignoring posts with other flairs.
    """
    
    #Create a fake Reddit submission with attributes:
    #flair, post id, post title, author, score, # of comments, timestamp, post's selftext, url
    mock_submission = MagicMock()
    mock_submission.link_flair_text = "Theory"
    mock_submission.id = "abc123"
    mock_submission.title = "Interesting Theory"
    mock_submission.author = "user1"
    mock_submission.score = 100
    mock_submission.num_comments = 10
    mock_submission.created_utc = 1234567890
    mock_submission.selftext = "This is the body"
    mock_submission.url = "http://example.com"


    #Create a fake subreddit that returns the fake submission
    mock_subreddit = MagicMock()
    mock_subreddit.search.return_value = [mock_submission]
    
    #Make the mocked Reddit API return the mocked subreddit just created
    mock_reddit.return_value.subreddit.return_value = mock_subreddit
    
    #Collector instantiation
    collector = RedditDataCollector("id", "secret", "agent", "OnePiece")
    
    #Run the method and collect posts from the mocked subreddit
    posts = collector.fetch_posts(["Theory"], limit=1)
    
    
    #Assertions:
    #lenght of posts collected is one    
    assert len(posts) == 1
    
    #post_id matching
    assert posts[0]["post_id"] == "abc123"
    
    #flair matching
    assert posts[0]["flair"] == "Theory"


@patch("praw.Reddit")
def test_fetch_posts_no_text(mock_reddit):
    
    """
    Given a mock subreddit with a post that an empty text,
    when the fetch function is called,
    then it should return the post and its 'selftext' attribute should be an empty string
    """
    
    #Create a fake Reddit submission with attributes:
    ##flair, post id, post title, author, score, # of comments, timestamp, post's selftext, url
    
    mock_submission = MagicMock()
    mock_submission.link_flair_text ="Theory"
    mock_submission.id = "qwe456"
    mock_submission.title = "A post with empty selftext"
    mock_submission.author = "user2"
    mock_submission.score = 80
    mock_submission.num_comments = 100
    mock_submission.created_utc = 12345678
    mock_submission.selftext = " "
    mock_submission.url = "https://example.com"
    
    #Create a fake subreddit that returns the fake submission
    mock_subreddit = MagicMock()
    mock_subreddit.search.return_value = [mock_submission]
    
    #Make the mocked Reddit API return the mocked subreddit just created
    mock_reddit.return_value.subreddit.return_value = mock_subreddit
    
    #Collector instantiation
    collector = RedditDataCollector("id","secret","agent","OnePiece")
    
    #Run the method and collect posts from the mocked subreddit
    posts = collector.fetch_posts(["Theory"], limit=1)
    
    #Assertions:
    #lenght of selftext is 1 (a blank space)
    assert len(posts[0]["selftext"]) == 1
    
    #the content of the post is a blank space
    assert posts[0]["selftext"] == ' '




