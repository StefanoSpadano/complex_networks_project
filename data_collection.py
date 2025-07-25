# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 12:36:11 2024

@author: Raffaele
"""
# -*- coding: utf-8 -*-
"""
Reddit Data Collection Script

This script collects posts and comments from a subreddit using Reddit API (PRAW).
It uses a hybrid configuration system:
- Sensitive credentials and defaults stored in config.ini
- Subreddit and filenames are dynamically generated
- Flairs can be specified interactively or via config/CLI
"""

import praw
import pandas as pd
import time
import random
import os
import warnings
import configparser
import argparse

#Suppress PRAW async warnings
warnings.filterwarnings("ignore", message="It appears that you are using PRAW in an asynchronous environment.")

#Load config.ini
config = configparser.ConfigParser()
config.read("config.ini")

#Define args = None so it exists globally

args = None

def prompt_user_for_flairs(flairs_in_posts, max_attempts=None):
    """
    Prompts user to select flairs by number or name.
    Keeps prompting until valid input is provided or max_attempts is reached.
    """
    attempts = 0
    while True:
        flairs_input = input("Enter flairs to include (comma separated, or press Enter for all): ").strip()

        if not flairs_input:
            return flairs_in_posts  #All flairs by default

        selected_flairs = []
        for entry in flairs_input.split(","):
            entry = entry.strip()
            if entry.isdigit():
                index = int(entry) - 1
                if 0 <= index < len(flairs_in_posts):
                    selected_flairs.append(flairs_in_posts[index])
                else:
                    print(f" Warning: {entry} is not a valid flair number.")
            else:
                matched_flair = next((f for f in flairs_in_posts if f.lower() == entry.lower()), None)
                if matched_flair:
                    selected_flairs.append(matched_flair)
                else:
                    print(f" Warning: '{entry}' is not a valid flair name.")

        if selected_flairs:
            return selected_flairs  #Valid selection

        print(" No valid flairs selected. Please try again.\n")
        attempts += 1
        if max_attempts is not None and attempts >= max_attempts:
            raise ValueError(" Too many invalid attempts. Exiting.")




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

    def fetch_posts(self, subreddit, target_flairs=None, limit=25):
        """
    Fetch posts from a subreddit, optionally filtering by flairs.

    Args:
        subreddit (str): Subreddit name
        target_flairs (list): List of flairs to filter by
        limit (int): Number of posts to fetch (default 25)

    Returns:
        list of dicts: Each dict represents a post
    """
        target_flairs = target_flairs or []
        subreddit_ref = self.reddit.subreddit(subreddit)
        posts = subreddit_ref.top(limit=limit)
        posts_data = []

        for post in posts:
            if target_flairs and post.link_flair_text not in target_flairs:
                continue  # Skip posts without desired flairs
            posts_data.append({
                "post_id": post.id,
                "title": post.title,
                "flair": post.link_flair_text,
                "score": post.score,
                "author": str(post.author),
                "created_utc": post.created_utc,
                "num_comments": post.num_comments,
                "url": post.url,
                "selftext": post.selftext
        })

        return posts_data


    def fetch_comments(self, post_id):
        """Fetch comments for a given post ID."""
        comments_data = []
        attempt = 0
        submission = self.reddit.submission(id=post_id)

        while attempt < 5:
            try:
                submission.comments.replace_more(limit=0)
                for comment in submission.comments.list()[:50]:
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

    def save_to_csv(self, data, filename):
        """Save a list of dictionaries to a CSV file."""
        
        import os
        #Ensure the folder exists
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Created folder: {folder}")
    
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        print(f"Saved data to {filename}")


def main():
    #Load credentials
    client_id = config["reddit"]["client_id"]
    client_secret = config["reddit"]["client_secret"]
    user_agent = config["reddit"]["user_agent"]

    #Subreddit (CLI > config > prompt)
    subreddit_name = args.subreddit or config["defaults"].get("subreddit")
    if not subreddit_name:
        subreddit_name = input("Enter subreddit to scrape: ").strip()
        if not subreddit_name:
            raise ValueError("Subreddit name is required.")
    
        #Flair logic (CLI > config > prompt)
    default_flairs = config["defaults"].get("flairs", "").split(",")
    target_flairs = (
        [f.strip() for f in args.flairs.split(",")] if args.flairs else
        [f.strip() for f in default_flairs if f.strip()]
    )

    
    #Generate default filenames based on subreddit
    subreddit_slug = subreddit_name.lower().replace(" ", "_")
    default_posts_file = f"../data/{subreddit_slug}_posts.csv"
    default_comments_file = f"../data/{subreddit_slug}_comments.csv"

        #Build dynamic filename if config is blank
    if config["defaults"].get("posts_output_file"):
        posts_file = config["defaults"]["posts_output_file"]
    else:
        posts_file = f"../data/{subreddit_slug}_posts.csv"
    
    if config["defaults"].get("comments_output_file"):
        comments_file = config["defaults"]["comments_output_file"]
    else:
        comments_file = f"../data/{subreddit_slug}_comments.csv"
    
    #Allow CLI args to override
    if args.posts_file:
        posts_file = args.posts_file
    if args.comments_file:
        comments_file = args.comments_file


    #Initialize collector
    collector = RedditDataCollector(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        subreddit_name=subreddit_name
    )

    if target_flairs:
        print(f" Using pre-configured flairs: {', '.join(target_flairs)}")
        #Fetch posts with pre-configured flairs
        posts_data = collector.fetch_posts(
            subreddit_name, target_flairs=target_flairs, limit=25
        )
    else:
        #Fetch posts first (to discover flairs dynamically)
        print(f"\nFetching posts from r/{subreddit_name}...")
        posts_data = collector.fetch_posts(
            subreddit_name, target_flairs=[], limit=25
        )
    
        #Discover unique flairs and prompt user
    from collections import Counter
    
    #Only prompt for flairs if none were pre-configured
    if not target_flairs:
        #Count occurrences of each flair
        flair_counts = Counter(post['flair'] for post in posts_data if post['flair'])
        flairs_in_posts = sorted(flair_counts.keys())
    
        if flairs_in_posts:
            print("\nFound these flairs in the top 25 posts:")
            for i, flair in enumerate(flairs_in_posts, 1):
                count = flair_counts[flair]
                print(f"{i}. {flair} ({count} posts)")
            #Prompt user ONCE for flair selection
            target_flairs = prompt_user_for_flairs(flairs_in_posts)
        else:
            print(" No flairs found in the top posts. Fetching all posts.")
            target_flairs = []  #No flair filtering

    
    #Filter posts again based on selected flairs
    if target_flairs:
        filtered_posts_data = [
            post for post in posts_data if post['flair'] in target_flairs
        ]
    else:
        filtered_posts_data = posts_data
    
    #Warn user if too few posts for meaningful analysis
    if len(filtered_posts_data) < 10:
        print(f"\n⚠️ Warning: Only {len(filtered_posts_data)} posts found after filtering.")
        print("👉 Consider selecting more flairs or increasing the post limit for better analysis results.")


    #Fetch comments
    comments_data = []
    print("\nFetching comments for each post...")
    for post in filtered_posts_data:
        comments_data.extend(collector.fetch_comments(post['post_id']))
        time.sleep(1)

    #Save data
    print("\nSaving collected data...")
    collector.save_to_csv(filtered_posts_data, posts_file)
    collector.save_to_csv(comments_data, comments_file)

    print(f"\n Data collection complete.\nPosts saved to {posts_file}\nComments saved to {comments_file}")




if __name__ == "__main__":
    #CLI argument parser
    parser = argparse.ArgumentParser(description="Reddit Data Collector")
    parser.add_argument("--subreddit", type=str, help="Subreddit to scrape")
    parser.add_argument("--flairs", type=str, nargs='+', help="List of flairs to filter by")
    parser.add_argument("--posts_file", type=str, help="Filename for saving posts data")
    parser.add_argument("--comments_file", type=str, help="Filename for saving comments data")
    args = parser.parse_args()
    main()






