# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:36:11 2024

@author: Raffaele
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:36:11 2024

@author: Raffaele
"""

import praw
import pandas as pd
import time
import random
import os
import warnings


# Suppress PRAW warning about asynchronous environment
warnings.filterwarnings("ignore", message="It appears that you are using PRAW in an asynchronous environment.")


from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

# Reddit API credentials
client_id = REDDIT_CLIENT_ID
client_secret = REDDIT_CLIENT_SECRET
user_agent = REDDIT_USER_AGENT


# Initialize Reddit client
def initialize_reddit_client(client_id, client_secret, user_agent):
    """
    Initializes and returns a Reddit API client instance.

    Args:
        client_id (str): Reddit API client ID.
        client_secret (str): Reddit API client secret.
        user_agent (str): User agent for the Reddit API.

    Returns:
        praw.Reddit: A Reddit API client instance.
    """
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent
    )


def fetch_posts(subreddit, target_flairs, limit=100):
    """
    Fetches posts from a subreddit that match the specified flairs.

    Args:
        subreddit (praw.models.Subreddit): The subreddit to fetch posts from.
        target_flairs (list): List of flairs to filter posts by.
        limit (int): Maximum number of posts to fetch.

    Returns:
        list: A list of dictionaries containing post data.
    """
    posts_data = []
    for submission in subreddit.search(query=" OR ".join(target_flairs), sort="top", limit=limit):
        if submission.link_flair_text in target_flairs:
            posts_data.append({
                'post_id': submission.id,
                'title': submission.title,
                'author': str(submission.author),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'selftext': submission.selftext,
                'url': submission.url,
                'flair': submission.link_flair_text
            })
    return posts_data


def fetch_comments(submission, post_id):
    """
    Fetches comments for a given post with exponential backoff.

    Args:
        submission (praw.models.Submission): The post to fetch comments for.
        post_id (str): The ID of the post.

    Returns:
        list: A list of dictionaries containing comment data.
    """
    comments_data = []
    attempt = 0
    while attempt < 5:  # Retry up to 5 times
        try:
            submission.comments.replace_more(limit=None)  # Load all comments
            for comment in submission.comments.list():
                comments_data.append({
                    'comment_id': comment.id,
                    'post_id': post_id,
                    'author': str(comment.author),
                    'body': comment.body,
                    'score': comment.score,
                    'created_utc': comment.created_utc
                })
            break  # Exit loop if successful

        except praw.exceptions.APIException as e:
            if 'RATELIMIT' in str(e):  # Check for rate limit error
                delay = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff with jitter
                print(f"Rate limit reached, retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                attempt += 1
            else:
                print(f"Unexpected API error: {e}")
                break  # Exit if it's not a rate limit error

        except Exception as e:
            print(f"Error fetching comments: {e}")
            break  # Exit on non-rate limit error

    return comments_data


def save_to_csv(data, file_path):
    """
    Saves data to a CSV file. Creates the directory if it doesn't exist.

    Args:
        data (list): A list of dictionaries to save.
        file_path (str): The path to save the CSV file.
    """
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def main():
    # Initialize Reddit client
    reddit = initialize_reddit_client(client_id, client_secret, user_agent)
    subreddit = reddit.subreddit("OnePiece")

    # Fetch posts and comments
    target_flairs = ["Theory", "Analysis", "Powerscaling"]
    posts_data = fetch_posts(subreddit, target_flairs)
    comments_data = []

    for post in posts_data:
        submission = reddit.submission(id=post['post_id'])
        comments_data.extend(fetch_comments(submission, post['post_id']))
        time.sleep(1)  # Short delay between posts

    # Save data to CSV files
    save_to_csv(posts_data, "../data/onepiece_posts.csv")
    save_to_csv(comments_data, "../data/onepiece_comments.csv")

    print("Data collection complete. Posts and comments have been saved.")


if __name__ == "__main__":
    main()





