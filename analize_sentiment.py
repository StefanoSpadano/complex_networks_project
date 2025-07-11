
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 11:32:48 2024

@author: Raffaele
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from utils import save_plot, categorize_sentiment, load_data




def calculate_sentiment(text, analyzer):
    """
    Calculate the sentiment score for a given text.

    Args:
        text (str): The text to analyze.
        analyzer (SentimentIntensityAnalyzer): The sentiment analyzer.

    Returns:
        float: The sentiment score (compound score from VADER).
    """
    if isinstance(text, str):  # Ensure text is a string
        sentiment_score = analyzer.polarity_scores(text)
        return sentiment_score['compound']
    else:
        return 0.0


def add_sentiment_to_posts(posts_df, analyzer):
    """
    Add sentiment scores to the posts DataFrame.

    Args:
        posts_df (pd.DataFrame): DataFrame containing posts data.
        analyzer (SentimentIntensityAnalyzer): The sentiment analyzer.

    Returns:
        pd.DataFrame: DataFrame with added sentiment scores.
    """
    posts_df['sentiment_selftext'] = posts_df['selftext'].apply(lambda x: calculate_sentiment(x, analyzer))
    return posts_df


def clean_sentiment_data(sentiment_posts):
    """
    Clean the sentiment data by removing rows with NaN values in 'selftext'.

    Args:
        sentiment_posts (pd.DataFrame): DataFrame containing sentiment data.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    sentiment_posts_cleaned = sentiment_posts.dropna(subset=['selftext'])
    return sentiment_posts_cleaned


def plot_sentiment_distribution(sentiment_posts_cleaned):
    """
    Plot the distribution of sentiment scores.

    Args:
        sentiment_posts_cleaned (pd.DataFrame): DataFrame containing cleaned sentiment data.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data=sentiment_posts_cleaned, x='sentiment_selftext', bins=20, kde=True, color='blue')
    plt.title('Distribution of Sentiment Scores', fontsize=14)
    plt.xlabel('Sentiment Score (-1 to 1)')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot("Distribution of sentiment scores", "plots/analize_sentiment_plots")
    plt.show()


def add_sentiment_category(sentiment_posts_cleaned):
    """
    Add a sentiment category column to the DataFrame.

    Args:
        sentiment_posts_cleaned (pd.DataFrame): DataFrame containing cleaned sentiment data.

    Returns:
        pd.DataFrame: DataFrame with added sentiment category.
    """
    sentiment_posts_cleaned['sentiment_category'] = sentiment_posts_cleaned['sentiment_selftext'].apply(categorize_sentiment)
    return sentiment_posts_cleaned



def plot_sentiment_categories(sentiment_posts_cleaned):
    """
    Plot the distribution of sentiment categories.

    Args:
        sentiment_posts_cleaned (pd.DataFrame): DataFrame containing sentiment categories.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(data=sentiment_posts_cleaned, x='sentiment_category', palette='coolwarm')
    plt.title('Distribution of Sentiment Categories', fontsize=14)
    plt.xlabel('Sentiment Category')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot("Distribution of sentiment categories", "plots/analize_sentiment_plots")
    plt.show()


def plot_flair_sentiment(sentiment_posts_cleaned):
    """
    Plot the average sentiment score by flair.

    Args:
        sentiment_posts_cleaned (pd.DataFrame): DataFrame containing cleaned sentiment data.
    """
    flair_sentiment = sentiment_posts_cleaned.groupby('flair')['sentiment_selftext'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=flair_sentiment, x='flair', y='sentiment_selftext', palette='coolwarm')
    plt.title('Average Sentiment Score by Flair', fontsize=14)
    plt.xlabel('Flair')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot("Average sentiment score by flair", "plots/analize_sentiment_plots")
    plt.show()


def plot_top_posts(sentiment_posts_cleaned):
    """
    Plot the most positive and negative posts.

    Args:
        sentiment_posts_cleaned (pd.DataFrame): DataFrame containing cleaned sentiment data.
    """
    top_positive_posts = sentiment_posts_cleaned.nlargest(5, 'sentiment_selftext')
    top_negative_posts = sentiment_posts_cleaned.nsmallest(5, 'sentiment_selftext')
    top_posts = pd.concat([top_positive_posts, top_negative_posts])

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_posts, x='sentiment_selftext', y='post_id', palette='coolwarm')
    plt.title('Most Positive and Negative Posts')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Post ID')
    save_plot("Most positive and negative posts", "plots/analize_sentiment_plots")
    plt.show()


def plot_author_sentiment(sentiment_posts_cleaned):
    """
    Plot the average sentiment per author.

    Args:
        sentiment_posts_cleaned (pd.DataFrame): DataFrame containing cleaned sentiment data.
    """
    author_sentiment = sentiment_posts_cleaned.groupby('author_x')['sentiment_selftext'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=author_sentiment, x='author_x', y='sentiment_selftext', palette='coolwarm')
    plt.title('Average Sentiment Per Author')
    plt.xlabel('Author')
    plt.ylabel('Average Sentiment Score')
    plt.xticks(rotation=90)
    save_plot("Average sentiment per author", "plots/analize_sentiment_plots")
    plt.show()


def plot_top_authors(sentiment_posts_cleaned):
    """
    Plot the top 10 authors by average sentiment.

    Args:
        sentiment_posts_cleaned (pd.DataFrame): DataFrame containing cleaned sentiment data.
    """
    author_sentiment = sentiment_posts_cleaned.groupby('author_x')['sentiment_selftext'].agg(['mean', 'count']).reset_index()
    author_sentiment = author_sentiment.sort_values(by='mean', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=author_sentiment.head(10), x='mean', y='author_x', palette='coolwarm')
    plt.title('Top 10 Authors by Average Sentiment')
    plt.xlabel('Average Sentiment Score')
    plt.ylabel('Author')
    save_plot("Top 10 authors by average sentiment", "plots/analize_sentiment_plots")
    plt.show()


def plot_sentiment_vs_engagement(sentiment_posts_cleaned):
    """
    Plot sentiment vs engagement (score).

    Args:
        sentiment_posts_cleaned (pd.DataFrame): DataFrame containing cleaned sentiment data.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=sentiment_posts_cleaned, x='score_x', y='sentiment_selftext', hue='flair', palette='viridis')
    plt.title('Score vs Sentiment')
    plt.xlabel('Score')
    plt.ylabel('Sentiment Score')
    plt.legend(title='Flair')
    save_plot("score vs sentiment", "plots/analize_sentiment_plots")
    plt.show()


def calculate_correlations(sentiment_posts_cleaned):
    """
    Calculate and print Pearson and Spearman correlations between sentiment and score.

    Args:
        sentiment_posts_cleaned (pd.DataFrame): DataFrame containing cleaned sentiment data.
    """
    pearson_corr, pearson_p = pearsonr(sentiment_posts_cleaned['sentiment_selftext'], sentiment_posts_cleaned['score_x'])
    spearman_corr, spearman_p = spearmanr(sentiment_posts_cleaned['sentiment_selftext'], sentiment_posts_cleaned['score_x'])
    print(f"Pearson correlation: {pearson_corr:.2f}, p-value: {pearson_p:.2e}")
    print(f"Spearman correlation: {spearman_corr:.2f}, p-value: {spearman_p:.2e}")


def save_cleaned_data(sentiment_posts_cleaned, output_path):
    """
    Save the cleaned sentiment data to a CSV file.

    Args:
        sentiment_posts_cleaned (pd.DataFrame): DataFrame containing cleaned sentiment data.
        output_path (str): Path to save the CSV file.
    """
    sentiment_posts_cleaned.to_csv(output_path, index=False)


def main():
    # Paths to data files
    posts_path = "../data/onepiece_posts.csv"
    post_metrics_path = "../data/post_metrics.csv"
    output_path = "../data/onepiece_sentiment_posts_filtered.csv"
    
    plt.ioff()  # Turn interactive mode off


    # Load data
    posts_df = load_data(posts_path)
    post_metrics =load_data(post_metrics_path)

    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Add sentiment to posts
    posts_df = add_sentiment_to_posts(posts_df, analyzer)

    # Merge post metrics and sentiment data
    sentiment_posts = posts_df.merge(post_metrics, on='post_id', how='inner')

    # Clean sentiment data
    sentiment_posts_cleaned = clean_sentiment_data(sentiment_posts)

    # Plot sentiment distribution
    plot_sentiment_distribution(sentiment_posts_cleaned)

    # Add sentiment category
    sentiment_posts_cleaned = add_sentiment_category(sentiment_posts_cleaned)

    # Plot sentiment categories
    plot_sentiment_categories(sentiment_posts_cleaned)

    # Plot flair sentiment
    plot_flair_sentiment(sentiment_posts_cleaned)

    # Plot top posts
    plot_top_posts(sentiment_posts_cleaned)

    # Plot author sentiment
    plot_author_sentiment(sentiment_posts_cleaned)

    # Plot top authors
    plot_top_authors(sentiment_posts_cleaned)

    # Plot sentiment vs engagement
    plot_sentiment_vs_engagement(sentiment_posts_cleaned)
    
    # Calculate correlations
    calculate_correlations(sentiment_posts_cleaned)

    # Save cleaned data
    save_cleaned_data(sentiment_posts_cleaned, output_path)
    USE_GLOBAL_SCOPE = False  # Set to True for Spyder's Variable Explorer

    if USE_GLOBAL_SCOPE:
       global df_posts, df_post_metrics, df_sentiment_posts, df_sentiment_clenaed
       df_posts = posts_df
       df_post_metrics = post_metrics
       df_sentiment_posts = sentiment_posts
       df_sentiment_cleaned = sentiment_posts_cleaned
    else:
       df_posts = posts_df
       df_post_metrics = post_metrics
       df_sentiment_posts = sentiment_posts
       df_sentiment_cleaned = sentiment_posts_cleaned



if __name__ == "__main__":
    main()
