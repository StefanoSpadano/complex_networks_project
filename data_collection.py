# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:36:11 2024

@author: Raffaele
"""


"""
This script collects posts and comments from a subreddit using Reddit API (PRAW),
with a hybrid configuration system (config.ini + CLI + interactive prompts).
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
    A class to collect posts and comments from a subreddit using PRAW.
    """
    def __init__(self, client_id, client_secret, user_agent, subreddit_name):
        self.reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        self.subreddit = self.reddit.subreddit(subreddit_name)

    def fetch_posts(self, target_flairs, limit=100):
        posts_data = []
        for submission in self.subreddit.search(query=" OR ".join(target_flairs), sort="top", limit=limit):
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

    def fetch_comments(self, post_id):
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
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)


def main():
    # Load credentials
    client_id = config["reddit"]["client_id"]
    client_secret = config["reddit"]["client_secret"]
    user_agent = config["reddit"]["user_agent"]

    # Subreddit (CLI > config > prompt)
    subreddit_name = args.subreddit or config["defaults"].get("subreddit") or input("Enter subreddit to scrape: ")

    # Flairs (CLI > config > prompt)
    default_flairs = config["defaults"].get("flairs", "").split(",")
    target_flairs = args.flairs or [flair.strip() for flair in default_flairs if flair.strip()] or \
        input("Enter flairs (comma separated): ").split(",")

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


# =============================================================================
# """
# This script is responsible for collecting posts and comments from the OnePiece subreddit
# using the Reddit API (PRAW). The data is then stored in CSV files for further analysis.
# The script is structured into a class-based design for possible modularity and reusability.
# """
# 
# import praw
# import pandas as pd
# import time
# import random
# import os
# import warnings
# 
# # Suppress PRAW warning about asynchronous environment
# warnings.filterwarnings("ignore", message="It appears that you are using PRAW in an asynchronous environment.")
# 
# from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT
# 
# 
# class RedditDataCollector:
#     """
#     A class to collect posts and comments from a given subreddit using PRAW.
#     """
#     def __init__(self, client_id, client_secret, user_agent, subreddit_name):
#         """
#         Initializes the RedditDataCollector instance with authentication details from the config.py file
#         and sets the target subreddit for data collection.
#         """
#         self.reddit = praw.Reddit(
#             client_id=client_id,
#             client_secret=client_secret,
#             user_agent=user_agent
#         )
#         self.subreddit = self.reddit.subreddit(subreddit_name)
# 
#     def fetch_posts(self, target_flairs, limit=100):
#         """
#         Fetches posts from the subreddit that match the specified flairs.
#         Uses a search query to retrieve posts that are relevant to the analysis.
#         
#         Args:
#             target_flairs (list): List of flairs to filter posts by. Available flairs are: 
#             Discussion, Theory, Powerscaling, Analysis, Fanart, Cosplay, Media, Merchandise, Big News.
#             limit (int): Maximum number of posts to fetch.
#         
#         Returns:
#             list: A list of dictionaries containing post data.
#         """
#         posts_data = []
#         for submission in self.subreddit.search(query=" OR ".join(target_flairs), sort="top", limit=limit):
#             if submission.link_flair_text in target_flairs:
#                 posts_data.append({
#                     'post_id': submission.id,
#                     'title': submission.title,
#                     'author': str(submission.author),
#                     'score': submission.score,
#                     'num_comments': submission.num_comments,
#                     'created_utc': submission.created_utc,
#                     'selftext': submission.selftext,
#                     'url': submission.url,
#                     'flair': submission.link_flair_text
#                 })
#         return posts_data
# 
#     def fetch_comments(self, post_id):
#         """
#         Fetches comments for a given post with exponential backoff to handle Reddit rate limits.
#         The function retries failed attempts with increasing wait times.
#         
#         Args:
#             post_id (str): The ID of the post for which comments are retrieved.
#         
#         Returns:
#             list: A list of dictionaries containing comment data.
#         """
#         comments_data = []
#         attempt = 0
#         submission = self.reddit.submission(id=post_id)
#         
#         while attempt < 5:  # Retry up to 5 times in case of rate limits
#             try:
#                 submission.comments.replace_more(limit=None)  # Load all comments
#                 for comment in submission.comments.list():
#                     comments_data.append({
#                         'comment_id': comment.id,
#                         'post_id': post_id,
#                         'author': str(comment.author),
#                         'body': comment.body,
#                         'score': comment.score,
#                         'created_utc': comment.created_utc
#                     })
#                 break  # Exit loop if successful
#             
#             except praw.exceptions.APIException as e:
#                 if 'RATELIMIT' in str(e):  # Check for rate limit error
#                     delay = 2 ** attempt + random.uniform(0, 1)  # Exponential backoff with jitter
#                     print(f"Rate limit reached, retrying in {delay:.2f} seconds...")
#                     time.sleep(delay)
#                     attempt += 1
#                 else:
#                     print(f"Unexpected API error: {e}")
#                     break  # Exit if it's not a rate limit error
#             
#             except Exception as e:
#                 print(f"Error fetching comments: {e}")
#                 break  # Exit on non-rate limit error
#         
#         return comments_data
# 
#     @staticmethod
#     def save_to_csv(data, file_path):
#         """
#         Saves the collected data to a CSV file. If the directory does not exist, it is created.
#         
#         Args:
#             data (list): A list of dictionaries containing the collected data.
#             file_path (str): The path to save the CSV file.
#         """
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         df = pd.DataFrame(data)
#         df.to_csv(file_path, index=False)
# 
# 
# def main():
#     """
#     Main function to initiate Reddit data collection.
#     Fetches posts and comments from the OnePiece subreddit and saves them to CSV files.
#     """
#     # Initialize data collector
#     collector = RedditDataCollector(
#         client_id=REDDIT_CLIENT_ID,
#         client_secret=REDDIT_CLIENT_SECRET,
#         user_agent=REDDIT_USER_AGENT,
#         subreddit_name="OnePiece"
#     )
# 
#     # Define the flairs of interest
#     target_flairs = ["Theory", "Analysis", "Powerscaling"]
#     
#     # Fetch posts
#     print("Fetching posts...")
#     posts_data = collector.fetch_posts(target_flairs)
#     
#     # Fetch comments for each post
#     comments_data = []
#     print("Fetching comments for each post...")
#     for post in posts_data:
#         comments_data.extend(collector.fetch_comments(post['post_id']))
#         time.sleep(1)  # Short delay between posts to avoid hitting API limits
#     
#     # Save collected data
#     print("Saving collected data...")
#     collector.save_to_csv(posts_data, "../data/onepiece_posts.csv")
#     collector.save_to_csv(comments_data, "../data/onepiece_comments.csv")
#     
#     print("Data collection complete. Posts and comments have been saved.")
# 
# 
# if __name__ == "__main__":
#     main()
# 
# =============================================================================




