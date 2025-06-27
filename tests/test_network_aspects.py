# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 18:00:53 2025

@author: Raffaele
"""

import networkx as nx
import pandas as pd

from network_aspects import assign_initial_sentiments, analyze_central_nodes

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
    
def test_assign_initial_sentiments_empty_dataframe_or_graph():
    """
    Given an empty dataframe or graph,
    when the assign_initial_sentiments function is called passing the empty dataframe or graph to it,
    then should return an empty dict and assign nothing.
    """
    #Initialize an empty graph
    empty_graph = nx.Graph()
    
    #Initialize an empty dataframe
    empty_df = pd.DataFrame(columns=["author", "sentiment_body"])
    
    #call the function
    assigned = assign_initial_sentiments(empty_df, empty_graph)
    
    #asserts
    #returns empty dict
    assert assigned == {}
    
    #no nodes have sentiment attribute
    assert all("sentiment" not in empty_graph.nodes[n] for n in empty_graph.nodes)


def test_analyze_central_nodes_identifies_high_degree_nodes():
    """
    Given a graph with varying degrees,
    when we want to analyze to 50% of the nodes calling the function analyze_central_nodes,
    then the degrees should be included.
    """
    #Initialize a graph with edges having various degress
    graph = nx.Graph()
    graph.add_edges_from([
        ("A", "B"), ("A", "C"), ("A", "D"), # A has degree 3
        ("B", "E"),                         # B has degree 2
        ("C", "F")                          # C has degree 2
                                            # D, E, F have degree 1
    ])

    #Call the function analyze_central_nodes to include the 50% percentile
    result = analyze_central_nodes(graph, percentile=50)
    
    #extract from result (a dictionary) the list of central_nodes
    central_nodes = result["central_nodes"]

    #asserts
    assert "A" in central_nodes
    assert isinstance(central_nodes, list)
    assert all(n in graph.nodes for n in central_nodes)


    
def test_analyze_central_nodes_returns_empty_on_empty_graph():
    """
    Given an empty graph as input,
    when the function analyze_central_nodes is called,
    then it should return an empty list.
   """
    #Initialize an empty graph
    graph = nx.Graph()

    #call the function on the empty graph
    result = analyze_central_nodes(graph)
    #extract the list of central nodes
    central_nodes = result["central_nodes"]

    #asserts
    assert isinstance(central_nodes, list)
    assert len(central_nodes) == 0
