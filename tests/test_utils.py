import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import categorize_sentiment, compute_flow_values, load_data
import numpy as np
import pandas as pd
import pytest

def test_load_data(tmp_path):
    """
    Given a path from where to load the data (in this case a temporary path),
    when the function load_data is called,
    then the correct dataframe should be displayed.
    """
    # Create a mock CSV file in a temporary path
    test_file = tmp_path / "test_data.csv"
    test_data = "col1,col2\n1,a\n2,b\n"
    test_file.write_text(test_data)

    # Run the function
    df = load_data(test_file)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)
    assert list(df.columns) == ['col1', 'col2']
    assert df.iloc[0]['col1'] == 1
    assert df.iloc[1]['col2'] == 'b'



def test_categorize_sentiment():
    """
    Given a set scores for sentiments,
    when the categorize sentiment function is called,
    then the correct sentiments (Positive, Negative and Neutral) should be returned.
    """
    # Test negative score
    assert categorize_sentiment(-0.5) == 'Negative'
    
    # Test borderline negative
    assert categorize_sentiment(-0.11) == 'Negative'
    
    # Test neutral score
    assert categorize_sentiment(0.0) == 'Neutral'
    
    # Test borderline positive
    assert categorize_sentiment(0.1) == 'Neutral'
    
    # Test positive score
    assert categorize_sentiment(0.5) == 'Positive'

def test_categorize_sentiment_edges():
    assert categorize_sentiment(-0.10000001) == 'Negative'
    assert categorize_sentiment(-0.09999999) == 'Neutral'
    assert categorize_sentiment(0.09999999) == 'Neutral'
    assert categorize_sentiment(0.10000001) == 'Positive'

def test_compute_flow_values():
    """
    Given a matrix containing numbers,
    when the function to compute flow values between rows is called,
    then the corresponding flows should be returned.
    """
    #define a matrixs
    matrix = np.array([
        [5, 2, 0],
        [1, 3, 4],
        [0, 0, 2]
    ])
    
    #call the function
    inter_flows, intra_flows = compute_flow_values(matrix)
    
    #asserts
    assert (0, 0, 5) not in inter_flows
    assert (1, 2, 4) in inter_flows
    assert (0, 5) in intra_flows
    assert len(inter_flows) == 3
    assert len(intra_flows) == 3

def test_compute_flow_values_edge_cases():
    """
    Given some different matrics covering edge cases,
    when the function to compute flow values is called,
    then the corresponding errors should be raised.
    """
    # Test 1: Empty matrix
    empty_matrix = np.array([[]])
    with pytest.raises(ValueError, match="empty"):
        compute_flow_values(empty_matrix)

    # Test 2: Non-square matrix (e.g., 2x3)
    nonsquare_matrix = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="square"):
        compute_flow_values(nonsquare_matrix)

    # Test 3: 1x1 matrix
    matrix_1x1 = np.array([[0.7]])
    inter_flows, intra_flows = compute_flow_values(matrix_1x1)
    assert inter_flows == []
    assert intra_flows == [(0, 0.7)]

    # Test 4: All-zero matrix
    zero_matrix = np.zeros((3, 3))
    inter_flows, intra_flows = compute_flow_values(zero_matrix)
    assert inter_flows == []
    assert intra_flows == [(0, 0.0), (1, 0.0), (2, 0.0)]
