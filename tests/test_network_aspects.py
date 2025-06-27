# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 18:00:53 2025

@author: Raffaele
"""

import networkx as nx
import pandas as pd

from network_aspects import assign_initial_sentiments  

def test_assign_initial_sentiments_classifies_mean_correctly():
    """
    Given a graph with some nodes and some comments with varying sentiment values belonging to different nodes,
    when the assign initial sentiments function is called,
    then the correct sentiment values should be assigned. 
    """
    #Initialize a graph with 3 nodes and comments with varying sentiment_body values
    graph = nx.Graph()
    graph.add_nodes_from(["user_pos", "user_neg", "user_neu"])
    
    #Initialize correspondance between users and their sentiment values
    data = {
        "author": ["user_pos", "user_pos", "user_neg", "user_neg", "user_neu"],
        "sentiment_body": [0.9, 0.8, -0.6, -0.7, 0.0]
    }
    df = pd.DataFrame(data)

    #Assigning initial sentiments
    assigned = assign_initial_sentiments(df, graph)

    #The correct sentiment values should be assigned
    assert assigned["user_pos"] == 1
    assert assigned["user_neg"] == -1
    assert assigned["user_neu"] == 0
    assert graph.nodes["user_pos"]["sentiment"] == 1
    assert graph.nodes["user_neg"]["sentiment"] == -1
    assert graph.nodes["user_neu"]["sentiment"] == 0

def test_assign_initial_sentiments_user_not_in_graph():
    """
    Given a missing user (with respect to those users in filtered_comments dataframe) in the graph,
    when the assign_initial_sentiments functions is called,
    then the function should skip them and/or don't assign anything.
    """
    #Initialize a graph with 3 nodes
    graph = nx.Graph()
    graph.add_nodes_from(["user_in_graph", "another_user_in_graph", "the_third_user_still_in_graph"])
    
    #Initialize a dataframe from which we are going to pick the sentiment values
    data = {
        "author": ["user_in_graph", "another_user_in_graph", "third_user_still_in_graph", "this_user_is_not_in_graph"],
        "sentiment_body": [0.9, 0.7, -0.4, 0.3]
        }
    df = pd.DataFrame(data)
    
    #Assigning initial sentiments
    assigned = assign_initial_sentiments(df, graph)
    
    #asserts
    assert len(assigned) == 3
    

def test_assign_initial_sentiments_user_neutral_and_polar_sentiments():
    """
    Given a user with a sentiment mix like [-0.5, 0.5, 0],
    when the assign_initial_sentiments function is called,
    then the function should return 0.
    """
    #Initialize a graph with a single node
    graph = nx.Graph()
    graph.add_nodes_from(["user_in_graph"])
    
    #Initialize a dataframe from which we pick the sentiment values
    data = {
        "author": ["user_in_graph", "user_in_graph", "user_in_graph"],
        "sentiment_body": [-0.5, 0.5, 0.0]
        }
    
    df = pd.DataFrame(data)
    
    #Call the function
    assigned = assign_initial_sentiments(df, graph)
    
    #asserts
    assert assigned["user_in_graph"] == 0.0