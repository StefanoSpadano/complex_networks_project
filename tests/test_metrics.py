import pandas as pd
from analize_metrics import calculate_unique_commenters

def test_unique_commenters_correct_counts_are_returned():
    """
    Given a set of Reddit posts and corresponding comments
    when we calculate the number of unique commenters for each post's author
    then the resulting dataframe should show the correct count of distinct commenters per author
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
    then the result should be an empty dataframe (no commenters found)
    """
    # GIVEN: A posts DataFrame with one post
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
