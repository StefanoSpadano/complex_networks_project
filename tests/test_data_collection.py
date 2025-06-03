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
    mock_subreddit.search.return_value = [mock_submission]
    
    #Make the mocked Reddit API return the mocked subreddit just created
    mock_reddit.return_value.subreddit.return_value = mock_subreddit
    
    #Collector instantiation
    collector = RedditDataCollector("id","secret","agent","OnePiece")
    
    #Run the method and collect posts from the mocked subreddit
    posts = collector.fetch_posts(["Theory"], limit=1)
    
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
    posts = collector.fetch_posts(["Theory"], limit=1)
    
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

def test_given_empty_data_when_saving_to_csv_then_create_csv_with_headers(tmp_path):
    """
    Given an empty data list
    when calling save_to_csv
    then it should create a CSV file with headers only and no data rows
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




def test_given_rate_limit_error_when_fetching_comments_then_retry_and_succeed():
    """
    Given a Reddit submission that raises a rate limit error initially
    when calling fetch_comments
    then the function should retry and eventually return the comment data
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
