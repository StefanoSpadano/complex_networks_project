import pandas as pd
import os
import tempfile
import pytest
from analize_metrics import calculate_unique_commenters, preprocess_data, calculate_post_metrics, save_metrics, calculate_comment_metrics

def test_unique_commenters_correct_counts_are_returned():
    """
    Given a set of Reddit posts and corresponding comments
    when we calculate the number of unique commenters for each post's author
    then the resulting dataframe should show the correct count of distinct commenters per author.
    """
    #Create a mock dataframe of posts with two different authors
    posts_data = {
        'post_id': [1, 2],
        'author': ['user1', 'user2']
    }
    posts_df = pd.DataFrame(posts_data)

    #create a mock dataframe of comments with different users commenting on each post
    comments_data = {
        'post_id': [1, 1, 2, 2, 2],  # Two comments on post 1, three on post 2
        'author': ['commenter1', 'commenter2', 'commenter1', 'commenter3', 'commenter1']
    }
    comments_df = pd.DataFrame(comments_data)

    #We compute unique commenters per post author
    result_df = calculate_unique_commenters(posts_df, comments_df)

    #Now the correct number of distinct commenters per author should be returned
    # user1 should have 2 unique commenters on post 1 (commenter1, commenter2)
    # user2 should have 2 unique commenters on post 2 (commenter1, commenter3)
    expected_data = {
        'author': ['user1', 'user2'],
        'unique_commenters': [2, 2]
    }
    expected_df = pd.DataFrame(expected_data)

    #We compare both DataFrames after sorting and resetting index to ignore row order
    pd.testing.assert_frame_equal(
        result_df.sort_values(by='author').reset_index(drop=True),
        expected_df.sort_values(by='author').reset_index(drop=True)
    )



def test_given_posts_with_empty_comments_then_zero_commenters_returned():
    """
    Given a set of Reddit posts and an empty comments dataframe
    when we calculate the number of unique commenters for each post's author
    then the result should be an empty dataframe (no commenters found).
    """
    #Initialize posts dataframe with one post
    posts_df = pd.DataFrame({
        'post_id': [1],
        'author': ['user1']
    })

    #Initialize an empty comments dataframe
    comments_df = pd.DataFrame(columns=['post_id', 'author'])

    #We calculate unique commenters
    result_df = calculate_unique_commenters(posts_df, comments_df)

    #The result should be an empty dataframe
    assert result_df.empty


def test_given_post_with_no_matching_comments_then_post_author_not_in_result():
    """
    Given a post whose post_id is not referenced in the comments
    when we calculate unique commenters
    then the author of that post should not appear in the results.
    """
    #Initialize a posts dataframe with post_id = 1
    posts_df = pd.DataFrame({
        'post_id': [1],
        'author': ['user1']
    })

    #Initialize comments dataframe referencing a different post_id (2 in this case)
    comments_df = pd.DataFrame({
        'post_id': [2, 2],
        'author': ['commenter1', 'commenter2']
    })

    #We calculate unique commenters
    result_df = calculate_unique_commenters(posts_df, comments_df)

    #At this point user1 should not be included in the result since no comments matched their post
    assert result_df.empty



def test_given_missing_values_when_preprocessing_then_drop_or_fill_as_expected():
    """
    Given posts and comments dataframes with missing values
    when we preprocess them
    then rows with missing author, score, or created_utc should be dropped
    and selftext should be filled with an empty string.
    """
    #Initialize a post dataframe with one valid row and two invalid rows
    posts_df = pd.DataFrame({
        'post_id': [1, 2, 3],
        'author': ['user1', None, 'user3'],
        'created_utc': [123, 456, None],
        'score': [10, None, 30],
        'selftext': ['valid post', None, 'another valid post']
    })

    #Initialize a comments dataframe with one valid row and one invalid row
    comments_df = pd.DataFrame({
        'comment_id': [101, 102],
        'post_id': [1, 2],
        'author': ['commenter1', None],
        'created_utc': [123, 456],
        'score': [5, 3],
        'body': ['text1', 'text2']
    })

    #Preprocessing the data with the function defined in analize_metrics.py
    cleaned_posts_df, cleaned_comments_df = preprocess_data(posts_df, comments_df)

    #Only one post should remain (the fully valid one)
    assert len(cleaned_posts_df) == 1
    assert cleaned_posts_df.iloc[0]['selftext'] == 'valid post'

    #Only one comment should remain, the one with a valid author
    assert len(cleaned_comments_df) == 1
    assert cleaned_comments_df.iloc[0]['author'] == 'commenter1'



def test_given_multiple_posts_when_calculating_post_metrics_then_correct_metrics_are_returned():
    """
    Given a DataFrame with multiple Reddit posts from different authors
    when we calculate post metrics
    then each post should contain its own info plus the author's average metrics.
    """
    #Initialize a posts dataframe by two authors with different scores and comment counts
    posts_df = pd.DataFrame({
        'post_id': [1, 2, 3],
        'author': ['user1', 'user1', 'user2'],
        'score': [10, 20, 30],
        'num_comments': [5, 15, 25]
    })

    #Calculate post metrics
    result_df = calculate_post_metrics(posts_df)

    #Verify metrics for one post per author
    user1_metrics = result_df[result_df['author'] == 'user1'].iloc[0]
    assert user1_metrics['total_upvotes'] in [10, 20]
    assert user1_metrics['average_upvotes_per_post'] == 15
    assert user1_metrics['total_posts'] == 2

    user2_metrics = result_df[result_df['author'] == 'user2'].iloc[0]
    assert user2_metrics['total_upvotes'] == 30
    assert user2_metrics['average_upvotes_per_post'] == 30
    assert user2_metrics['total_posts'] == 1




def test_save_metrics_saves_csv_with_expected_columns():
    """
    Given a sample dataframe containing post and comment metrics and a file path
    when the function is called
    then both files should exist and contain the expected columns.
    """
    
    #Initialize the sample dataframe with post metrics attributes
    post_metrics = pd.DataFrame({
        'post_id': [1],
        'author': ['user1'],
        'score': [10],
        'num_comments': [5],
        'total_upvotes': [10],
        'total_comments': [5],
        'post_count': [1],
        'average_upvotes_per_post': [10.0],
        'average_comments_per_post': [5.0],
        'total_posts': [1]
    })
    #Initialize a sample data frame with comment author and its total comments
    comment_metrics = pd.DataFrame({
        'author': ['commenter1'],
        'total_comments': [3]
    })

    #Create temporary file paths
    with tempfile.TemporaryDirectory() as tmp_dir:
        post_path = os.path.join(tmp_dir, "post_metrics.csv")
        comment_path = os.path.join(tmp_dir, "comment_metrics.csv")

        #Call the save_metrics function
        save_metrics(post_metrics, comment_metrics, post_path, comment_path)

        #At this point both files should exist and contain expected columns
        saved_post_df = pd.read_csv(post_path)
        saved_comment_df = pd.read_csv(comment_path)
        
        #Asserts:
        #Check expected column names
        assert list(saved_post_df.columns) == list(post_metrics.columns)
        assert list(saved_comment_df.columns) == list(comment_metrics.columns)



def test_save_metrics_raises_error_on_invalid_path():
    """
    Given a dataframe with minimal attributes
    when saved into an invalid file path
    then should raise the correspondent error.
    """
    
    #Initialize two dataframes one for post_metrics and one for comment_metrics
    post_metrics = pd.DataFrame({'author': ['user1'], 'score': [10]})
    comment_metrics = pd.DataFrame({'author': ['user2'], 'total_comments': [2]})

    #Initialize two invalid file paths
    invalid_post_path = "/invalid_directory/post_metrics.csv"
    invalid_comment_path = "/invalid_directory/comment_metrics.csv"

    #Here calling save_metrics should raise an IOError (OSError / FileNotFoundError)
    with pytest.raises((FileNotFoundError, OSError)):
        save_metrics(post_metrics, comment_metrics, invalid_post_path, invalid_comment_path)



def test_post_with_no_comments_should_be_excluded_from_unique_commenters():
    """
    Given a post with no comments,
    when calculate_unique_commenters is called,
    then the resulting dataframe should be empty (no authors returned).
    """
    #Initialize a dataframe with one post by 'user1' and no comments
    posts_df = pd.DataFrame({'post_id': [1], 'author': ['user1']})
    comments_df = pd.DataFrame(columns=['post_id', 'author'])  #Empty comment DataFrame

    #Calculating unique commenters
    result_df = calculate_unique_commenters(posts_df, comments_df)

    #No rows should be returned, since no commenters exist
    assert result_df.empty
    assert list(result_df.columns) == ['author', 'unique_commenters']




def test_multiple_posts_same_author_aggregates_unique_commenters():
    """
    Given two posts from the same author and multiple unique commenters,
    when calling the calculate_unique_commenters function,
    then the correct number of commenters per author should be returned.
    """
    #Initialize a dataframe containing two posts by the same author and multiple unique commenters
    posts_df = pd.DataFrame({
        'post_id': [1, 2],
        'author': ['user1', 'user1']  # Same author for both posts
    })
    comments_df = pd.DataFrame({
        'post_id': [1, 1, 2, 2],
        'author': ['commenter1', 'commenter2', 'commenter3', 'commenter1']  #'commenter1' is repeated
    })

    #Calculating unique commenters
    result_df = calculate_unique_commenters(posts_df, comments_df)

    #The author 'user1' should have 3 unique commenters in total
    expected_df = pd.DataFrame({'author': ['user1'], 'unique_commenters': [3]})
    pd.testing.assert_frame_equal(result_df.sort_values(by='author').reset_index(drop=True),
                                  expected_df.sort_values(by='author').reset_index(drop=True))



def test_post_with_zero_score_and_comments_returns_metrics():
    """
    Given a dataframe with only one post that has zero comments and zero score,
    when calling the function calculate_unique_commenters,
    then the result should still be returned. 
    """
    #Initialize a dataframe with one post that has zero score and comments
    posts_df = pd.DataFrame({
        'post_id': [1],
        'author': ['user1'],
        'score': [0], #0 score
        'num_comments': [0] #0 comments
    })

    #Calculating post metrics
    result_df = calculate_post_metrics(posts_df)

    #Asserts:
    #Metrics should still be returned and not raise an error
    assert result_df.shape[0] == 1
    assert result_df.loc[0, 'total_upvotes'] == 0
    assert result_df.loc[0, 'total_comments'] == 0
    assert result_df.loc[0, 'average_upvotes_per_post'] == 0



def test_empty_comments_df_returns_empty_metrics():
    """
    Given a comments dataframe that is empty (no comments),
    when we calculate the dataframe metrics,
    then an empty dataframe should be returned.
    """
    #Initialize an empty comments dataframe
    comments_df = pd.DataFrame(columns=['author', 'body'])

    #Calculating comment metrics
    result_df = calculate_comment_metrics(comments_df)

    #The result should also be an empty DataFrame
    assert result_df.empty
    assert list(result_df.columns) == ['author', 'total_comments']




def test_duplicate_authors_are_aggregated_correctly():
    """
    Given a dataframe storing multiple comments from the same user,
    when calling the calculate_comment_metrics function,
    then the comments for the same user should be summed up.
    """
    #Initialize a comments dataframe with user that commented multiple times
    comments_df = pd.DataFrame({
        'author': ['user1', 'user1', 'user2'],
        'body': ['a', 'b', 'c']
    })

    #Calculating comment metrics
    result_df = calculate_comment_metrics(comments_df)

    #'user1' should have 2 comments, 'user2' should have 1
    expected_df = pd.DataFrame({
        'author': ['user1', 'user2'],
        'total_comments': [2, 1]
    })
    pd.testing.assert_frame_equal(result_df.sort_values(by='author').reset_index(drop=True),
                                  expected_df.sort_values(by='author').reset_index(drop=True))
