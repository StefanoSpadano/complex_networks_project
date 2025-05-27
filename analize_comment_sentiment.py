
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:45:01 2024

@author: Raffaele
"""

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import powerlaw
from scipy.stats import pearsonr, spearmanr

from utils import save_plot, categorize_sentiment



def load_data(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, lineterminator='\n', engine='python')

# =============================================================================
# def load_comments_data(comments_path):
#     """
#     Load comments data from a CSV file.
# 
#     Args:
#         comments_path (str): Path to the comments CSV file.
# 
#     Returns:
#         pd.DataFrame: DataFrame containing comments data.
#     """
#     comments_df = pd.read_csv(comments_path)
#     return comments_df
# 
# =============================================================================

def filter_comments(comments_df):
    """
    Filter comments to remove deleted authors, empty bodies, and low engagement scores.

    Args:
        comments_df (pd.DataFrame): DataFrame containing comments data.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    filtered_comments = comments_df[
        (comments_df['author'].notna()) &
        (comments_df['author'] != '[deleted]') &
        (comments_df['body'].str.strip().astype(bool)) &
        (comments_df['score'] > 0)
    ]
    return filtered_comments


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


def add_sentiment_to_comments(filtered_comments, analyzer):
    """
    Add sentiment scores to the comments DataFrame.

    Args:
        filtered_comments (pd.DataFrame): DataFrame containing filtered comments data.
        analyzer (SentimentIntensityAnalyzer): The sentiment analyzer.

    Returns:
        pd.DataFrame: DataFrame with added sentiment scores.
    """
    filtered_comments['sentiment_body'] = filtered_comments['body'].apply(lambda x: calculate_sentiment(x, analyzer))
    return filtered_comments


def plot_sentiment_distribution(filtered_comments):
    """
    Plot the distribution of sentiment scores.

    Args:
        filtered_comments (pd.DataFrame): DataFrame containing filtered comments data.
    """
    plt.figure(figsize=(10, 6))
    sns.kdeplot(filtered_comments['sentiment_body'], shade=True)
    plt.title('Kernel Density Estimate of Sentiment Scores')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Density')
    save_plot("Kernel Density Estimate of Sentiment Scores","plots/analize_comment_sentiment_plots")
    plt.show()


# =============================================================================
# def categorize_sentiment(sentiment_score):
#     """
#     Categorize a sentiment score into 'Positive', 'Neutral', or 'Negative'.
# 
#     Args:
#         sentiment_score (float): The sentiment score.
# 
#     Returns:
#         str: The sentiment category.
#     """
#     if sentiment_score > 0:
#         return 'Positive'
#     elif sentiment_score < 0:
#         return 'Negative'
#     else:
#         return 'Neutral'
# =============================================================================


def add_sentiment_category(filtered_comments):
    """
    Add a sentiment category column to the DataFrame.

    Args:
        filtered_comments (pd.DataFrame): DataFrame containing filtered comments data.

    Returns:
        pd.DataFrame: DataFrame with added sentiment category.
    """
    filtered_comments['sentiment_category'] = filtered_comments['sentiment_body'].apply(categorize_sentiment)
    return filtered_comments


def plot_engagement_by_sentiment(filtered_comments):
    """
    Plot the average engagement by sentiment category.

    Args:
        filtered_comments (pd.DataFrame): DataFrame containing filtered comments data.
    """
    engagement_by_category = (
        filtered_comments.groupby('sentiment_category')['score']
        .mean()
        .reset_index()
        .sort_values('score', ascending=False)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(
        engagement_by_category['sentiment_category'],
        engagement_by_category['score'],
        color=['green', 'gray', 'red']
    )
    plt.title('Average Engagement by Sentiment Category')
    plt.xlabel('Sentiment Category')
    plt.ylabel('Average Engagement (Score)')
    save_plot("Average Engagement by Sentiment Category","plots/analize_comment_sentiment_plots")
    plt.show()


def plot_comment_scores(filtered_comments):
    """
    Plot the distribution of comment scores.

    Args:
        filtered_comments (pd.DataFrame): DataFrame containing filtered comments data.
    """
    comment_scores = filtered_comments['score']

    # Regular histogram
    plt.figure(figsize=(10, 5))
    plt.hist(comment_scores, bins=50, color='skyblue', edgecolor='black')
    plt.title('Regular Histogram of Comment Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    save_plot("Regular Histogram of Comment Scores","plots/analize_comment_sentiment_plots")
    plt.show()

    # Log-log histogram
    log_bins = np.logspace(np.log10(comment_scores.min() + 1), np.log10(comment_scores.max() + 1), 50)
    plt.figure(figsize=(10, 5))
    plt.hist(comment_scores, bins=log_bins, color='skyblue', edgecolor='black', log=True)
    plt.yscale('log')
    plt.title('Log-Log Histogram of Comment Scores')
    plt.xlabel('Score (Log Scale)')
    plt.ylabel('Frequency (Log Scale)')
    save_plot("Log-Log Histogram of Comment Scores","plots/analize_comment_sentiment_plots")
    plt.show()


def fit_power_law(filtered_comments):
    """
    Fit the comment scores to a power-law distribution and plot the results.

    Args:
        filtered_comments (pd.DataFrame): DataFrame containing filtered comments data.
    """
    fit = powerlaw.Fit(filtered_comments['score'], xmin=1)
    print("Power-Law Alpha:", fit.alpha)
    print("xmin (Cutoff):", fit.xmin)

    plt.figure(figsize=(10, 6))
    fit.plot_ccdf(color='blue', label='Empirical Data')
    fit.power_law.plot_ccdf(color='red', linestyle='--', label='Fitted Power-Law')
    plt.title('CCDF of Comment Scores')
    plt.xlabel('Score')
    plt.ylabel('CCDF')
    plt.legend()
    save_plot("CCDF of Comment Scores","plots/analize_comment_sentiment_plots")
    plt.show()


def compare_distributions(filtered_comments):
    """
    Compare power-law and lognormal fits for comment scores.

    Args:
        filtered_comments (pd.DataFrame): DataFrame containing filtered comments data.
    """
    data = filtered_comments['score'].values
    powerlaw_fit = powerlaw.Fit(data, xmin=1, discrete=True)
    lognorm_fit = powerlaw.Fit(data, xmin=1, discrete=True)
    lognorm_mu = lognorm_fit.lognormal.mu
    lognorm_sigma = lognorm_fit.lognormal.sigma

    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.6, color='gray', label='Data (Engagement Scores)')

    x = np.linspace(1, max(data), 1000)
    lognorm_pdf = (1 / (x * lognorm_sigma * np.sqrt(2 * np.pi))) * \
                  np.exp(-((np.log(x) - lognorm_mu) ** 2) / (2 * lognorm_sigma ** 2))
    plt.plot(x, lognorm_pdf, color='blue', linewidth=2, label='Lognormal Fit')

    alpha = powerlaw_fit.alpha
    xmin = powerlaw_fit.xmin
    x_powerlaw = np.linspace(xmin, max(data), 1000)
    powerlaw_pdf = (x_powerlaw ** (-alpha)) / np.sum(x_powerlaw ** (-alpha))
    plt.plot(x_powerlaw, powerlaw_pdf, color='red', linestyle='--', linewidth=2, label='Power-law Fit')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Engagement Scores (log scale)')
    plt.ylabel('Density (log scale)')
    plt.title('Comparison of Lognormal and Power-law Fits')
    plt.legend()
    save_plot("Comparison of Lognormal and Power-law Fits","plots/analize_comment_sentiment_plots")
    plt.show()

    print("Lognormal mu (mean):", lognorm_fit.lognormal.mu)
    print("Lognormal sigma (std):", lognorm_fit.lognormal.sigma)

    R, p = lognorm_fit.distribution_compare('lognormal', 'power_law')
    print(f"Loglikelihood ratio (R): {R}")
    print(f"P-value of comparison: {p}")


def analyze_top_comments(filtered_comments, N=100):
    """
    Analyze the top N comments by engagement score.

    Args:
        filtered_comments (pd.DataFrame): DataFrame containing filtered comments data.
        N (int): Number of top comments to analyze.
    """
    top_comments = filtered_comments.sort_values(by="score", ascending=False).head(N)
    top_summary = top_comments[['comment_id', 'score', 'body', 'sentiment_body']].copy()
    top_summary['length'] = top_summary['body'].apply(len)

    print("Top Comments Summary:")
    print(top_summary)

    # Save for further analysis
    top_summary.to_csv("top_comments_summary.csv", index=False)

    # Plot engagement vs. sentiment
    plt.figure(figsize=(10, 6))
    plt.scatter(top_summary['sentiment_body'], top_summary['score'], color='blue', alpha=0.7)
    plt.title('Engagement vs. Sentiment for Top Comments')
    plt.xlabel('Sentiment (Compound Score)')
    plt.ylabel('Engagement Score')
    plt.grid()
    save_plot("Engagement vs. Sentiment for Top Comments","plots/analize_comment_sentiment_plots")
    plt.show()

    # Plot length vs. sentiment
    plt.figure(figsize=(10, 6))
    plt.scatter(top_summary['length'], top_summary['sentiment_body'], color='green', alpha=0.7)
    plt.title('Length vs. Sentiment for Top Comments')
    plt.xlabel('Length of Comment')
    plt.ylabel('Sentiment (Compound Score)')
    plt.grid()
    save_plot("Length vs. Sentiment for Top Comments","plots/analize_comment_sentiment_plots")
    plt.show()

    # Calculate Pearson correlation
    correlation, p_value = pearsonr(top_summary['length'], top_summary['sentiment_body'])
    print(f"Pearson Correlation: {correlation:.4f}, P-value: {p_value:.4e}")

    # Calculate Spearman correlation
    spearman_corr, p_value = spearmanr(top_summary['length'], top_summary['sentiment_body'])
    print(f"Spearman Correlation: {spearman_corr:.4f}, P-value: {p_value:.4e}")
    
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(
        x=top_summary['length'],
        y=top_summary['sentiment_body'],
        cmap='Blues',
        fill=True,
        ax=ax
    )
    ax.set_title('KDE Heatmap: Length vs. Sentiment')
    ax.set_xlabel('Length of Comment')
    ax.set_ylabel('Sentiment (Compound Score)')
    ax.grid(True)

    fig.tight_layout()
    fig.savefig("plots/analize_comment_sentiment_plots/KDE Heatmap - Length vs Sentiment.png")
    plt.close(fig)


# =============================================================================
#     # KDE heatmap
#     plt.figure(figsize=(10, 6))
#     sns.kdeplot(
#         x=top_summary['length'],
#         y=top_summary['sentiment_body'],
#         cmap='Blues',
#         fill=True
#     )
#     plt.title('KDE Heatmap: Length vs. Sentiment')
#     plt.xlabel('Length of Comment')
#     plt.ylabel('Sentiment (Compound Score)')
#     plt.grid()
#     #plt.tight_layout()  # <-- Important in order to save correctly the plot in the folder as with sns works differently
#     save_plot("KDE Heatmap: Length vs. Sentiment","plots/analize_comment_sentiment_plots")
#     plt.savefig("debug_kde.png")
#     plt.show()
# 
# =============================================================================


def save_filtered_comments(filtered_comments, output_path):
    """
    Save the filtered comments data to a CSV file.

    Args:
        filtered_comments (pd.DataFrame): DataFrame containing filtered comments data.
        output_path (str): Path to save the CSV file.
    """
    filtered_comments.to_csv(output_path, index=False)


def main():
    # Paths to data files
    comments_path = "../data/onepiece_comments.csv"
    output_path = "../data/onepiece_sentiment_comments_filtered.csv"

    # Load comments data
    comments_df = load_data(comments_path)

    # Filter comments
    filtered_comments = filter_comments(comments_df)

    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Add sentiment to comments
    filtered_comments = add_sentiment_to_comments(filtered_comments, analyzer)

    # Plot sentiment distribution
    plot_sentiment_distribution(filtered_comments)

    # Add sentiment category
    filtered_comments = add_sentiment_category(filtered_comments)

    # Plot engagement by sentiment
    plot_engagement_by_sentiment(filtered_comments)

    # Plot comment scores
    plot_comment_scores(filtered_comments)

    # Fit power-law distribution
    fit_power_law(filtered_comments)

    # Compare distributions
    compare_distributions(filtered_comments)

    # Analyze top comments
    analyze_top_comments(filtered_comments)

    # Save filtered comments
    save_filtered_comments(filtered_comments, output_path)
    USE_GLOBAL_SCOPE = False  # Set to True for Spyder's Variable Explorer
    
    if USE_GLOBAL_SCOPE:
       global df_posts, df_post_metrics, df_sentiment_posts, df_sentiment_clenaed
       df_comments = comments_df
       df_filtered_comments = filtered_comments
    else:
       df_comments = comments_df
       df_filtered_comments = filtered_comments

   


if __name__ == "__main__":
    main()