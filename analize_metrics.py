# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:24:17 2024

@author: Raffaele
"""

import configparser
import argparse
import os
import pandas as pd
from utils import load_data


def preprocess_data(posts_df, comments_df):
    """
    Preprocess the posts and comments data.

    Args:
        posts_df (pd.DataFrame): DataFrame containing posts data.
        comments_df (pd.DataFrame): DataFrame containing comments data.

    Returns:
        tuple: A tuple containing two DataFrames (posts_df, comments_df).
    """
    #Preprocessing for posts_df by removing missing values inside these columns
    posts_df.dropna(subset=['author', 'created_utc', 'score'], inplace=True)
    # Fill missing post content with an empty string
    posts_df['selftext'] = posts_df['selftext'].fillna('')

    #Preprocessing for comments_df by removing missing values inside these columns
    comments_df.dropna(subset=['author', 'created_utc', 'score'], inplace=True)

    return posts_df, comments_df


def calculate_post_metrics(posts_df):
    """
    Calculate post-level metrics.

    Args:
        posts_df (pd.DataFrame): DataFrame containing posts data.

    Returns:
        pd.DataFrame: DataFrame containing post-level metrics.
    """
    #Create a DataFrame specifically for post-level metrics using 'post_id' and 'author'
    post_metrics = posts_df[['post_id', 'author', 'score', 'num_comments']].copy()

    #Calculate total metrics by post
    post_metrics['total_upvotes'] = post_metrics['score']
    post_metrics['total_comments'] = post_metrics['num_comments']
    post_metrics['post_count'] = 1  # Each row is a single post

    #Calculate average metrics per post by author
    author_metrics = post_metrics.groupby('author').agg(
        average_upvotes_per_post=('total_upvotes', 'mean'),
        average_comments_per_post=('total_comments', 'mean'),
        total_posts=('post_count', 'sum')
    ).reset_index()

    #Merge to retain author-level insights
    post_metrics = post_metrics.merge(author_metrics, on='author', how='left')

    return post_metrics


def calculate_unique_commenters(posts_df, comments_df):
    """
    Calculate the number of unique commenters for each post author.

    Args:
        posts_df (pd.DataFrame): DataFrame containing posts data.
        comments_df (pd.DataFrame): DataFrame containing comments data.

    Returns:
        pd.DataFrame: DataFrame containing unique commenter counts.
    """
    #Group commenters by post ID
    commenters_dict = comments_df.groupby('post_id')['author'].apply(set).to_dict()

    #Calculate unique commenters for each post author
    author_commenters = {}
    for post_id, commenters in commenters_dict.items():
        #Identify post author based on post_id in posts_df
        post_author = posts_df.loc[posts_df['post_id'] == post_id, 'author'].values
        if post_author.size > 0:  #Ensure post author is found
            post_author = post_author[0]
            if post_author not in author_commenters:
                author_commenters[post_author] = set()
            author_commenters[post_author].update(commenters)

    #Convert to DataFrame for merging
    unique_comment_counts = pd.DataFrame(
        [(author, len(commenters)) for author, commenters in author_commenters.items()],
        columns=['author', 'unique_commenters']
    )

    return unique_comment_counts


def calculate_comment_metrics(comments_df):
    """
    Calculate comment-level metrics.

    Args:
        comments_df (pd.DataFrame): DataFrame containing comments data.

    Returns:
        pd.DataFrame: DataFrame containing comment-level metrics.
    """
    #Comment metrics for users who commented
    comment_metrics = comments_df.groupby('author').agg(
        total_comments=('body', 'size')
    ).reset_index()

    return comment_metrics


def save_metrics(post_metrics, comment_metrics, post_metrics_path, comment_metrics_path):
    """
    Save the post and comment metrics to CSV files.

    Args:
        post_metrics (pd.DataFrame): DataFrame containing post-level metrics.
        comment_metrics (pd.DataFrame): DataFrame containing comment-level metrics.
        post_metrics_path (str): Path to save the post metrics CSV file.
        comment_metrics_path (str): Path to save the comment metrics CSV file.
    """
    post_metrics.to_csv(post_metrics_path, index=False)
    comment_metrics.to_csv(comment_metrics_path, index=False)


def main():
    #Load config and CLI arguments
    config = configparser.ConfigParser()
    config.read("config.ini")

    parser = argparse.ArgumentParser(description="Analyze subreddit metrics")
    parser.add_argument("--subreddit", type=str, help="Subreddit to analyze")
    parser.add_argument("--posts_file", type=str, help="Path to posts CSV")
    parser.add_argument("--comments_file", type=str, help="Path to comments CSV")
    parser.add_argument("--post_metrics_file", type=str, help="Path to save post metrics CSV")
    parser.add_argument("--comment_metrics_file", type=str, help="Path to save comment metrics CSV")
    args = parser.parse_args()
    
        #Subreddit (CLI > config > prompt)
    subreddit_name = args.subreddit or config["defaults"].get("subreddit")
    if not subreddit_name:
        subreddit_name = input("Enter subreddit to analyze: ").strip()
        if not subreddit_name:
            raise ValueError("Subreddit name is required.")
    subreddit_slug = subreddit_name.lower().replace(" ", "_")
        
    #Dynamic filenames based on subreddit + flairs
    default_posts_file = f"../data/{subreddit_slug}_posts.csv"
    default_comments_file = f"../data/{subreddit_slug}_comments.csv"
    default_post_metrics_file = f"../data/{subreddit_slug}_post_metrics.csv"
    default_comment_metrics_file = f"../data/{subreddit_slug}_comment_metrics.csv"
    
    posts_path = args.posts_file or config["defaults"].get("posts_output_file") or default_posts_file
    comments_path = args.comments_file or config["defaults"].get("comments_output_file") or default_comments_file
    post_metrics_path = args.post_metrics_file or config["defaults"].get("post_metrics_output_file") or default_post_metrics_file
    comment_metrics_path = args.comment_metrics_file or config["defaults"].get("comments_metrics_output_file") or default_comment_metrics_file
    
    print(f" Using posts file: {posts_path}")
    print(f" Using comments file: {comments_path}")

    
    #Load data
    posts_df = load_data(posts_path)
    comments_df = load_data(comments_path)

    #Preprocess data
    posts_df, comments_df = preprocess_data(posts_df, comments_df)

    #Calculate post metrics
    post_metrics = calculate_post_metrics(posts_df)

    #Calculate unique commenters
    unique_comment_counts = calculate_unique_commenters(posts_df, comments_df)

    #Merge unique commenters with post metrics
    post_metrics = post_metrics.merge(unique_comment_counts, on='author', how='left')
    post_metrics['unique_commenters'].fillna(0, inplace=True)

    #Calculate comment metrics
    comment_metrics = calculate_comment_metrics(comments_df)

    #Save metrics
    save_metrics(post_metrics, comment_metrics, post_metrics_path, comment_metrics_path)

    #Debugging: Inspect the data
    print("Post Metrics:")
    print(post_metrics.head())

    print("\nComment Metrics:")
    print(comment_metrics.head())

    print("Metrics DataFrames saved.")


if __name__ == "__main__":
    main()














