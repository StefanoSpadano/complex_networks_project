# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:36:11 2024

@author: Raffaele
"""

import praw
import pandas as pd
import time
import random


# Reddit API credentials
client_id = '95411n-TS3tjNZnPjhNCAA'
client_secret = 'ROi0S-firoCE042IQKdvhvLzu2LIgA'
user_agent = 'OnePieceScraper by /u/yourusername'

# Create Reddit instance (without username and password)
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     user_agent=user_agent)

# Set the subreddit and the query parameters
subreddit = reddit.subreddit("OnePiece")

# Define dataframes for posts and comments
posts_data = []
comments_data = []

# Function to fetch posts with specific flairs and their comments with exponential backoff
def fetch_filtered_posts_and_comments():
    target_flairs = ["Theory", "Analysis", "Powerscaling"]
    
    for submission in subreddit.search(query=" OR ".join(target_flairs), sort="top", limit=100):
        # Check if the flair matches one of the target flairs
        if submission.link_flair_text in target_flairs:
            post_id = submission.id
            posts_data.append({
                'post_id': post_id,
                'title': submission.title,
                'author': str(submission.author),
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_utc': submission.created_utc,
                'selftext': submission.selftext,
                'url': submission.url,
                'flair': submission.link_flair_text
            })

            # Fetch comments for the post with exponential backoff
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

            # Adding a short delay after each post to prevent hitting limits too quickly
            time.sleep(1)


# Run the function to collect data
fetch_filtered_posts_and_comments()

# Save posts and comments data to CSV files
posts_df = pd.DataFrame(posts_data)
comments_df = pd.DataFrame(comments_data)

posts_df.to_csv("../data/onepiece_posts.csv", index=False)
comments_df.to_csv("../data/onepiece_comments.csv", index=False)

print("Data collection complete. Posts and comments have been saved.")






