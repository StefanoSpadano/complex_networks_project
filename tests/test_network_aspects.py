# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 18:00:53 2025

@author: Raffaele
"""

import networkx as nx
import pandas as pd

from network_aspects import (assign_initial_sentiments, analyze_central_nodes, compute_mini_cluster_centrality,
                             calculate_sentiment_influence, compute_sentiment_flow_matrix)


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

def test_analyze_central_nodes_nodes_having_equal_degree():
    """
    Given a graph as input with nodes habing all the same degree,
    when the analyze_central_nodes function is called,
    then it should return all or none according to threshold set.
    """
    #Initialize a graph
    graph = nx.Graph()
    
    #add edges so that all nodes have the same degree
    graph.add_edges_from([
        ("A","B"),("B", "C"),("C", "A")
    ])
    
    #call the function
    result = analyze_central_nodes(graph, percentile=100)
    #extract the list of central nodes
    central_nodes = result["central_nodes"]
    
    #asserts
    assert set(central_nodes) == {"A", "B", "C"}


def test_compute_mini_cluster_centrality_on_triangle_graph():
    """
    Given a graph with some nodes and edges between nodes,
    when we call the function to compute centrality measures,
    then we get correct values from the centralities we are interested in.
    """
    #Initialize a graph, a triangle graph here for semplicity
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])

    #Call the function to compute centralities (in a mini cluster in my case)
    df = compute_mini_cluster_centrality(graph)

    #Assert that all nodes should have the same centrality values
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"degree", "betweenness", "closeness"}
    assert set(df.index) == {"A", "B", "C"}

    #All degree centralities should be equal and equal to 1.0 (because of the way we defined our graph from the start)
    assert all(df["degree"] == 1.0)

    #All closeness centralities should be equal and between 0 and 1
    closeness_vals = df["closeness"].unique()
    assert len(closeness_vals) == 1
    assert 0 < closeness_vals[0] <= 1

    #Betweenness should be zero in a fully connected triangle
    assert all(df["betweenness"] == 0)

def test_compute_mini_cluster_centrality_disconnected_graph():
    """
    Given a graph in which we have a node which is not connected to anything,
    when we call the compute mini cluster centrality function,
    then we expect for that node to have a closeness and betweenness centrality of zero.
    """
    #Initialize a graph
    graph = nx.Graph()
    graph.add_edges_from([
        ("A", "B"),  ("B", "A")
        ])
    #We have to explicitly define a disconnected node
    graph.add_node("C")
    
    #Call the function to compute mini cluster centralities
    df = compute_mini_cluster_centrality(graph)
    
    #Asserts
    #Access the corresponding row of the second column
    assert df["betweenness"].iloc[2] == 0
    #Another way to access the closeness value of the disconnected node
    assert df.loc["C", "closeness"] == 0
    
def test_compute_mini_cluster_centrality_single_node_graph():
    """
    Given a graph with a single node and no connections,
    when the compute centrality function is called,
    then it should return zero for both closeness and betweenness.
    """
    #Initialize a graph
    graph = nx.Graph()
    #add a node
    graph.add_node("A")
    
    #call the function
    df = compute_mini_cluster_centrality(graph)
    
    #Asserts
    assert df.loc["A", "betweenness"] == 0
    assert df.loc["A", "closeness"] == 0


def test_sentiment_influence_counts_shared_neighbors():
    """
    Given a graph with known sentiments associated to each node,
    when calculating sentiment influence,
    then the correct influences should be returned.
    """
    #Initialize a graph and add edges
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("A", "C"), ("A", "D"), ("B", "C")])
    #set node sentiments for each node
    nx.set_node_attributes(G, {
        "A": {"sentiment": 1},
        "B": {"sentiment": 1},
        "C": {"sentiment": -1},
        "D": {"sentiment": 1}
    })

    #Calculating sentiment influence
    result = calculate_sentiment_influence(G)

    #Node A has 2 neighbors with same sentiment (B and D), 1 different (C)
    assert result["A"] == 2
    assert result["B"] == 1  # only A
    assert result["C"] == 0  # all neighbors are different
    assert result["D"] == 1  # only A

def test_sentiment_influence_isolated_node():
    """
    Given a graph with an isolated node,
    when calculating the sentiment influence,
    then its influence score should be zero. 
    """
    #Initialize a graph
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    #initialize an isolated node
    graph.add_node("D")
    #set node sentiments
    nx.set_node_attributes(graph, {
        "A": {"sentiment":1},
        "B": {"sentiment": -1},
        "C": {"sentiment": 1},
        "D": {"sentiment": 1}
        })
    
    #calculate sentiment influence
    result = calculate_sentiment_influence(graph)
    
    #assert
    assert result["D"] == 0
    
def test_sentiment_influence_all_neighbors_have_same_influence():
    """
    Given a graph with nodes having all the same sentiment,
    when the calculate sentiment influence function is called,
    then the influence score should be equal to the number of nodes in the graph.
    """
    #Initialize a graph
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    
    #set the nodes to have all the same sentiment
    nx.set_node_attributes(graph, {
        "A": {"sentiment": 1},
        "B": {"sentiment": 1},
        "C": {"sentiment": 1}
        })
    
    #call the function
    result = calculate_sentiment_influence(graph)
    
    #asserts
    #sentiment influence should be 2 because one node doesnt influence itself
    assert result["A"] == result ["B"] == result["C"] == 2

def test_sentiment_influence_no_attribute_present():
    """
    Given a graph with sentiment attributes in each node except one,
    when the function calculate_sentiment_infuence is called,
    then return zero or skip it. 
    """
    #initialize a graph
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "A")])
    
    #set node attributes
    nx.set_node_attributes(graph, {
        "A": {"sentiment": 1},
        "B": {"sentiment": 1}
        })
    
    #call the function
    result = calculate_sentiment_influence(graph)
    
    #asserts
    assert result["C"] == 0
    

def test_sentiment_flow_matrix_basic_case():
    """
    Given nodes connected with some sentimentes,
    when the compute sentiment flow matrix is called,
    then it should return the correct flow of sentiments (sentiments are summed if they are aligned).
    """
    #Initialize a graph
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("A", "C")])
    
    #assign nodes to different communities
    partition = {"A": 0, "B": 1, "C": 1}
    #assign sentiments to each node
    nx.set_node_attributes(G, {"A": 1, "B": 1, "C": -1}, name="sentiment")
    
    #call the function
    matrix = compute_sentiment_flow_matrix(G, partition)
    #print to check
    print("Matrix:\n", matrix)

    # Only A–B should be counted, same sentiment
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] == 1.0
    assert matrix[1, 0] == 1.0
    assert matrix[0, 0] == 0.0
    assert matrix[1, 1] == 0.0
    
def test_sentiment_flow_matrix_edges_different_sentiment():
    """
    Given a graph where nodes A and B are in different communities and they have different sentiments,
    when the function compute sentiment flow matrix is called,
    then it should return a matrix of all zeros (no shared sentiment).
    """
    #Initialize a graph
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("A", "C")])
    
    #assign nodes to different communities
    partition = {"A":0, "B":1, "C":1}
    
    #assign sentiments to the nodes
    nx.set_node_attributes(graph, {"A": 1, "B":-1, "C":-1}, name = "sentiment")
    
    #call the function that returns a matrix with flow values from different communities
    matrix = compute_sentiment_flow_matrix(graph, partition)
    
    #asserts
    assert matrix.shape == (2,2)
    assert matrix [0,1] == 0.0
    assert matrix [0,0] == 0.0
    assert matrix [1,0] == 0.0
    assert matrix [1,1] == 0.0
    
def test_sentiment_flow_matrix_missing_sentiment_on_one_node():
    """
    Given a graph where node A has a sentiment and node B has no sentiment at all,
    when computing the matrix of sentiment flow values,
    then the edge with missing sentiment should not be included.
    """
    #initialize a graph
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("A", "C")])
    
    #assign nodes to different communities
    partition = {"A":0, "B":1, "C":1}
    
    #assign sentiments to nodes
    nx.set_node_attributes(graph, {
        "A":1, "C":-1
        }, name="sentiment")
    
    #call the function
    matrix = compute_sentiment_flow_matrix(graph, partition)
    
    #asserts
    assert matrix.shape == (2,2)
    assert matrix [0,1] == 0.0
    assert matrix [0,0] == 0.0
    assert matrix [1,0] == 0.0
    assert matrix [1,1] == 0.0
    
def test_sentiment_flow_matrix_wighted_graph():
    """
    Given a graph with nodes A and B having the same sentiment and the edge A-B has a weight different than 1,
    when computing the sentiment flow matrix,
    then we expect that the matrix has the corresponding value in the corresponding place.
    """
    #initialize a graph
    graph = nx.Graph()
    graph.add_edges_from([("A", "B"), ("A", "C")])
    
    #assign nodes to different communities
    partition = {"A":0, "B":1, "C":1}
    
    #assign node sentiment
    nx.set_node_attributes(graph, {
        "A":1, "B":1, "C":-1
        }, name = "sentiment")
    
    #modify the weight for edge A-B
    graph["A"]["B"]["weight"] = 5

    
    #call the function
    matrix = compute_sentiment_flow_matrix(graph, partition)
    #asserts
    assert matrix.shape == (2,2)
    assert matrix [0,1] == 5.0
    assert matrix [1,0] == 5.0
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



