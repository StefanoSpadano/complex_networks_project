import pandas as pd
from analize_metrics import calculate_unique_commenters

def test_calculate_unique_commenters():
    # Mock data for posts
    posts_data = {
        'post_id': [1, 2],
        'author': ['user1', 'user2']
    }
    posts_df = pd.DataFrame(posts_data)

    # Mock data for comments
    comments_data = {
        'post_id': [1, 1, 2, 2, 2],
        'author': ['commenter1', 'commenter2', 'commenter1', 'commenter3', 'commenter1']
    }
    comments_df = pd.DataFrame(comments_data)

    # Run the function
    result_df = calculate_unique_commenters(posts_df, comments_df)

    # Expected result
    expected_data = {
        'author': ['user1', 'user2'],
        'unique_commenters': [2, 2]  # user1 has 2 commenters on post 1, user2 has 2 on post 2
    }
    expected_df = pd.DataFrame(expected_data)

    # Assert equality ignoring row order
    pd.testing.assert_frame_equal(result_df.sort_values(by='author').reset_index(drop=True),
                                  expected_df.sort_values(by='author').reset_index(drop=True))

