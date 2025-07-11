# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 16:57:50 2025

@author: Raffaele
"""

import tempfile
import os
from unittest.mock import patch
import praw
from unittest.mock import patch, MagicMock
from complex_networks_project.data_collection import RedditDataCollector, prompt_user_for_flairs
import argparse
import sys
import os

# Add the parent folder (project root) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import data_collection



#Temporarily replaces praw.Reddit class just for test function
@patch("praw.Reddit")
def test_fetch_posts_return_matching_flair(mock_reddit):
    """
    Given a subreddit with posts of various flairs,
    when fetch_posts is called with a target flair,
    then it should return only posts matching that flair.
    """
    mock_reddit_instance = mock_reddit.return_value
    mock_subreddit = mock_reddit_instance.subreddit.return_value

    # Mock post with matching flair
    mock_post = MagicMock()
    mock_post.id = "abc123"
    mock_post.title = "Test Post"
    mock_post.link_flair_text = "Theory"  # 
    mock_post.score = 100
    mock_post.author = "test_user"
    mock_post.created_utc = 1234567890
    mock_post.num_comments = 5
    mock_post.url = "https://reddit.com/abc123"
    mock_post.selftext = "Test content"

    mock_subreddit.top.return_value = [mock_post]

    collector = RedditDataCollector(
        client_id="dummy_id",
        client_secret="dummy_secret",
        user_agent="dummy_agent",
        subreddit_name="OnePiece"
    )

    posts = collector.fetch_posts("OnePiece", target_flairs=["Theory"], limit=1)

    assert len(posts) == 1
    assert posts[0]["flair"] == "Theory"



@patch("praw.Reddit")
def test_fetch_posts_no_text(mock_reddit):
    
    """
    Given a mock subreddit with a post that has an empty text,
    when the fetch function is called,
    then it should return the post and its 'selftext' attribute should be an empty string.
    """
    
    #Create a fake Reddit submission with attributes:
    #flair, post id, post title, author, score, # of comments, timestamp, post's selftext, url
    
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
    mock_subreddit.top.return_value = [mock_submission]
    
    #Make the mocked Reddit API return the mocked subreddit just created
    mock_reddit.return_value.subreddit.return_value = mock_subreddit
    
    #Collector instantiation
    collector = RedditDataCollector("id","secret","agent","OnePiece")
    
    #Run the method and collect posts from the mocked subreddit
    posts = collector.fetch_posts("OnePiece", target_flairs=["Theory"], limit=1)
    
    #Assertions:
    #lenght of selftext is 1 (a blank space)
    assert len(posts[0]["selftext"]) == 1
    
    #the content of the post is a blank space
    assert posts[0]["selftext"] == ' '

@patch("praw.Reddit")
def test_fetch_posts_missing_fields(mock_reddit):
    
    """
    Given a mock subreddit with a post containing a missin field in the attributes,
    when the fetch function is called,
    then it should return the value None for that attribute.
    """
    
    #Createa fake Reddit submission with attributes:
    ##flair, post id, post title, author, score, # of comments, timestamp, post's selftext, url
    
    mock_submission = MagicMock()
    mock_submission.link_flair_text ="Theory"
    mock_submission.id = "qwe456"
    mock_submission.title = "A post with empty selftext"
    mock_submission.author = None #simulates deletion of this attribute
    mock_submission.score = 80
    mock_submission.num_comments = 100
    mock_submission.created_utc = 12345678
    mock_submission.selftext = " "
    mock_submission.url = "https://example.com"
    
    #Create a fake subreddit that returns the fake submission
    mock_subreddit = MagicMock()
    mock_subreddit.top.return_value = [mock_submission]
    
    #Make the mocked Reddit API return the mocked subreddit just created
    mock_reddit.return_value.subreddit.return_value = mock_subreddit
    
    #Collector instantiation
    collector = RedditDataCollector("id","secret","agent","OnePiece")
    
    #Run the method and collect posts from the mocked subreddit
    posts = collector.fetch_posts("OnePiece", target_flairs=["Theory"], limit=1)
    
    #Assertions:
    #content of the missing field is None
    assert posts[0]["author"] == 'None'
    
@patch("praw.Reddit")
def test_fetch_posts_different_flair(mock_reddit):
    
    """
    Given a mock subreddit with a post containing a post belonging to a different flair with
    respect to those specified,
    when the fetch function is called,
    then it should return no posts at all.
    """
    
    #Createa fake Reddit submission with attributes:
    ##flair, post id, post title, author, score, # of comments, timestamp, post's selftext, url
    
    mock_submission = MagicMock()
    mock_submission.link_flair_text ="Another_Flair"
    mock_submission.id = "ijn098"
    mock_submission.title = "A post of an unrelated flair"
    mock_submission.author = "Unrelated flair author" 
    mock_submission.score = 0
    mock_submission.num_comments = 5
    mock_submission.created_utc = 2345678
    mock_submission.selftext = "This post has an unrelated flair"
    mock_submission.url = "https://example.com"
    
    #Create a fake subreddit that returns the fake submission
    mock_subreddit = MagicMock()
    mock_subreddit.search.return_value = [mock_submission]
    
    #Make the mocked Reddit API return the mocked subreddit just created
    mock_reddit.return_value.subreddit.return_value = mock_subreddit
    
    #Collector instantiation
    collector = RedditDataCollector("id","secret","agent","OnePiece")
    
    #Run the method and collect posts from the mocked subreddit
    posts = collector.fetch_posts("OnePiece", target_flairs=["Theory"], limit=1)
    
    #Assertions:
    #verify that no posts has been collected
    assert len(posts) == 0 
    

@patch('praw.Reddit')
def test_fetch_comments_returns_expected_data(mock_reddit):
    """
    Given a submission with two comments,
    when the function fetch_comments is called,
    then it should return the correct data extracted from both comments.
    """
    
    #Create a fake submission's comment with attributes:
    #id, author, body, score, timestamp
    mock_comment1 = MagicMock()
    mock_comment1.id = 'c1'
    mock_comment1.author = 'User1'
    mock_comment1.body = 'First comment'
    mock_comment1.score = 12
    mock_comment1.created_utc = 1681000000
    
    #Create a second fake sumbision's comment with attributes:
    #id, author, body, score, timestamp
    mock_comment2 = MagicMock()
    mock_comment2.id = 'c2'
    mock_comment2.author = 'User2'
    mock_comment2.body = 'Second comment'
    mock_comment2.score = 8
    mock_comment2.created_utc = 1681000100
    
    #Create a fake submission that returns the same comments just mocked
    mock_submission = MagicMock()
    mock_submission.comments.replace_more.return_value = None
    mock_submission.comments.list.return_value = [mock_comment1, mock_comment2]
    
    #Make the mocked Reddit API return the mocked submission just created
    mock_reddit.return_value.submission.return_value = mock_submission
    
    #Instantiation
    collector = RedditDataCollector('id', 'secret', 'agent', 'OnePiece')
    
    #Run the method and collect comments from the mocked submission
    comments = collector.fetch_comments('abc123')
    
    
    #Assertions:
    #verify that two comments were collected
    assert len(comments) == 2
    #verify that the comment_id of the first comment correspondes to 'c1'
    assert comments[0]['comment_id'] == 'c1'
    #verify correspondance between second comment and its body
    assert comments[1]['body'] == 'Second comment'
    



def test_save_to_csv_creates_file():
    """
    Given a list of dictionaries as data and a temporary file path,
    when the save_to_csv function is called,
    then it should create a CSV file at the specified location with the mimicked contents.
    """

    # Sample data that mimics what might be returned from Reddit API
    sample_data = [
        {'id': '1', 'text': 'Hello'},
        {'id': '2', 'text': 'World'}
    ]

    # Create a temporary directory to write files on
    with tempfile.TemporaryDirectory() as tmpdir:
        # Construct a full file path inside the temporary directory
        filepath = os.path.join(tmpdir, 'test_output.csv')

        # Call the static method to save data to CSV
        RedditDataCollector.save_to_csv(sample_data, filepath)

        # Check if the file now exists at the specified location
        assert os.path.exists(filepath), "CSV file was not created"

        # Open the file and read its lines to verify content
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # The first line should be the CSV header
            assert 'id,text' in lines[0], "CSV header is incorrect"

            # Check that the rows correspond to the sample data
            assert '1,Hello' in lines[1], "First row does not match"
            assert '2,World' in lines[2], "Second row does not match"

def test_empty_data_when_saving_to_csv(tmp_path):
    """
    Given an empty data list
    when calling save_to_csv
    then it should create a CSV file with headers only and no data rows.
    """
    # Arrange: Define empty data and file path
    empty_data = []
    file_path = tmp_path / "empty.csv"

    # Act: Attempt to save empty data
    RedditDataCollector.save_to_csv(empty_data, file_path)

    # Assert: File is created and contains only headers (i.e., empty file or just newline)
    with open(file_path, 'r') as f:
        content = f.read().strip()
    
    assert content == ""  # Since the data is empty, even headers are unknown (no fields to infer)




def test_rate_limit_error():
    """
    Given a Reddit submission that raises a rate limit error initially
    when calling fetch_comments
    then the function should retry and eventually return the comment data.
    """
    # Arrange
    collector = RedditDataCollector("id", "secret", "agent", "subreddit")

    # Create a mock comment object
    mock_comment = MagicMock()
    mock_comment.id = "comment123"
    mock_comment.author = "user123"
    mock_comment.body = "test comment"
    mock_comment.score = 10
    mock_comment.created_utc = 999999

    # Mock submission with replace_more raising error once, then succeeding
    mock_submission = MagicMock()
    mock_submission.comments.replace_more.side_effect = [
        praw.exceptions.APIException("RATELIMIT", "ratelimit message", None),
        None  # Second attempt succeeds
    ]
    mock_submission.comments.list.return_value = [mock_comment]

    # Patch the Reddit API to return our mock submission
    with patch.object(collector.reddit, 'submission', return_value=mock_submission):
        # Act
        comments = collector.fetch_comments("post123")

    # Assert: Should succeed on second attempt and fetch 1 comment
    assert len(comments) == 1
    assert comments[0]['comment_id'] == "comment123"

def test_prompt_user_for_flairs_number_selection(monkeypatch):
    """
    Given a list of flairs and user input of numbers,
    when prompt_user_for_flairs is called,
    then it should return the selected flairs matching those numbers.
    """
    flairs_in_posts = ["Theory", "Discussion", "Fanart"]
    user_input = "1,3"  # User selects "Theory" and "Fanart"

    #Simulate user input
    monkeypatch.setattr("builtins.input", lambda _: user_input)

    selected_flairs = prompt_user_for_flairs(flairs_in_posts)

    assert selected_flairs == ["Theory", "Fanart"]

def test_prompt_user_for_flairs_user_selects_all_flairs(monkeypatch):
    """
    Given a list of flairs, 
    when prompt_user_for_flairs is called and the user does not enter anything,
    then all flairs should be selected by default.
    """
    flairs_in_posts = ["Theory", "Discussion", "Fanart"]
    
    #Simulate the input
    monkeypatch.setattr("builtins.input", lambda _: "")
    
    selected_flairs = prompt_user_for_flairs(flairs_in_posts)
    
    assert selected_flairs == ["Theory","Discussion", "Fanart"]

def test_prompt_user_for_flairs_invalid_numbers(monkeypatch):
    """
    Given a list of flairs and user input of invalid numbers,
    when prompt_user_for_flairs is called with max_attempts=1,
    then it should raise a ValueError.
    """
    flairs_in_posts = ["Theory", "Discussion", "Fanart"]

    # Simulate user entering invalid numbers
    monkeypatch.setattr("builtins.input", lambda _: "10,20")

    try:
        prompt_user_for_flairs(flairs_in_posts, max_attempts=1)
        assert False, "Expected ValueError for too many invalid attempts"
    except ValueError as e:
        assert "Too many invalid attempts" in str(e)

def test_prompt_user_for_flairs_case_insensitive(monkeypatch):
    """
    Given a list of flairs,
    when the function prompt_user_for_flairs is called and flairs are inserted in lowercase,
    then the function should still match them correctly.
    """
    flairs_in_posts = ["Theory", "Discussion", "Fanart"]
    
    #Simulate the input
    monkeypatch.setattr("builtins.input", lambda _: "theory, discussion")
    
    selected_flairs = prompt_user_for_flairs(flairs_in_posts, max_attempts=1)
    assert selected_flairs == ["Theory", "Discussion"]
    
def test_prompt_user_for_flairs_mix_number_names(monkeypatch):
    """
    Given a list of flairs,
    when the function prompt_user_for_flairs is called and the user inserts both numbers and flairs' names,
    then the function should still match them correctly.
    """
    flairs_in_posts = ["Theory", "Discussion", "Fanart"]
    
    #Simulate the input
    monkeypatch.setattr("builtins.input", lambda _: "1, fanart")
    
    selected_flairs = prompt_user_for_flairs(flairs_in_posts, max_attempts=1)
    
    assert selected_flairs == ["Theory", "Fanart"]

