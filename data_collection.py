# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:36:11 2024

@author: Raffaele
"""


"""
This script collects posts and comments from a subreddit using Reddit API (PRAW),
with a hybrid configuration system (config.ini + CLI + interactive prompts).
"""

# -*- coding: utf-8 -*-
"""
Reddit Data Collection Script (Hybrid Config)

This script collects posts and comments from a subreddit using Reddit API (PRAW).
It uses a hybrid configuration system:
- Sensitive credentials and defaults stored in config.ini
- Subreddit, flair, and filenames can be overridden via CLI or interactive prompts
"""

import praw
import pandas as pd
import time
import random
import os
import warnings
import configparser
import argparse

# Suppress PRAW async warnings
warnings.filterwarnings("ignore", message="It appears that you are using PRAW in an asynchronous environment.")

# Load config.ini
config = configparser.ConfigParser()
config.read("config.ini")

# CLI argument parser
parser = argparse.ArgumentParser(description="Reddit Data Collector")
parser.add_argument("--subreddit", type=str, help="Subreddit to scrape")
parser.add_argument("--flairs", type=str, nargs='+', help="List of flairs to filter by")
parser.add_argument("--posts_file", type=str, help="Filename for saving posts data")
parser.add_argument("--comments_file", type=str, help="Filename for saving comments data")
args = parser.parse_args()


class RedditDataCollector:
    """
    RedditDataCollector provides methods for fetching posts and comments
    from a subreddit using Reddit's API via PRAW.

    Parameters
    ----------
    client_id : str
        Reddit API client ID from your Reddit app (config.ini).
    client_secret : str
        Reddit API client secret (config.ini).
    user_agent : str
        Descriptive string identifying your app to Reddit API.
    subreddit_name : str
        The name of the subreddit to scrape.

    Raises
    ------
    ValueError
        If subreddit_name is empty.
    """
    def __init__(self, client_id, client_secret, user_agent, subreddit_name):
        if not subreddit_name:
            raise ValueError("Subreddit name cannot be empty.")

        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.subreddit = self.reddit.subreddit(subreddit_name)

    def fetch_posts(self, target_flairs, limit=100):
        """Fetch posts from the subreddit filtered by flair."""
        posts_data = []
        for submission in self.subreddit.hot(limit=limit):
            if target_flairs and submission.link_flair_text not in target_flairs:
                continue  # Skip posts without desired flair
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

    def fetch_comments(self, post_id):
        """Fetch comments for a given post ID."""
        comments_data = []
        attempt = 0
        submission = self.reddit.submission(id=post_id)

        while attempt < 5:
            try:
                submission.comments.replace_more(limit=None)
                for comment in submission.comments.list():
                    comments_data.append({
                        'comment_id': comment.id,
                        'post_id': post_id,
                        'author': str(comment.author),
                        'body': comment.body,
                        'score': comment.score,
                        'created_utc': comment.created_utc
                    })
                break
            except praw.exceptions.APIException as e:
                if 'RATELIMIT' in str(e):
                    delay = 2 ** attempt + random.uniform(0, 1)
                    print(f"Rate limit reached, retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    attempt += 1
                else:
                    print(f"Unexpected API error: {e}")
                    break
            except Exception as e:
                print(f"Error fetching comments: {e}")
                break

        return comments_data

    @staticmethod
    def save_to_csv(data, file_path):
        """Save data to a CSV file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)


def main():
    # Load credentials
    client_id = config["reddit"]["client_id"]
    client_secret = config["reddit"]["client_secret"]
    user_agent = config["reddit"]["user_agent"]

    # Subreddit (CLI > config > prompt)
    subreddit_name = args.subreddit or config["defaults"].get("subreddit")
    if not subreddit_name:
        subreddit_name = input("Enter subreddit to scrape: ").strip()
        if not subreddit_name:
            raise ValueError("Subreddit name is required.")

    # Flairs (CLI > config > prompt)
    default_flairs = config["defaults"].get("flairs", "").split(",")
    target_flairs = args.flairs or [flair.strip() for flair in default_flairs if flair.strip()]
    if not target_flairs:
        flairs_input = input("Enter flairs (comma separated, leave blank for all): ").strip()
        target_flairs = [f.strip() for f in flairs_input.split(",")] if flairs_input else []

    # Output filenames (CLI > config > defaults)
    posts_file = args.posts_file or config["defaults"].get("posts_output_file", "posts.csv")
    comments_file = args.comments_file or config["defaults"].get("comments_output_file", "comments.csv")

    # Initialize collector
    collector = RedditDataCollector(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        subreddit_name=subreddit_name
    )

    # Fetch posts
    print("Fetching posts...")
    posts_data = collector.fetch_posts(target_flairs)

    # Fetch comments
    comments_data = []
    print("Fetching comments for each post...")
    for post in posts_data:
        comments_data.extend(collector.fetch_comments(post['post_id']))
        time.sleep(1)

    # Save data
    print("Saving collected data...")
    collector.save_to_csv(posts_data, posts_file)
    collector.save_to_csv(comments_data, comments_file)

    print(f"Data collection complete.\nPosts saved to {posts_file}\nComments saved to {comments_file}")


if __name__ == "__main__":
    main()





