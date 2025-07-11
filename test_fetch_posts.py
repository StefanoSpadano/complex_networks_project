# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:57:50 2025

@author: Raffaele
"""

import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from data_collection import RedditDataCollector  


@patch("praw.Reddit")
def test_fetch_posts(mock_reddit):
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

    mock_subreddit = MagicMock()
    mock_subreddit.search.return_value = [mock_submission]
    mock_reddit.return_value.subreddit.return_value = mock_subreddit

    collector = RedditDataCollector("id", "secret", "agent", "OnePiece")
    posts = collector.fetch_posts(["Theory"], limit=1)

    assert len(posts) == 1
    assert posts[0]["post_id"] == "abc123"
    assert posts[0]["flair"] == "Theory"
