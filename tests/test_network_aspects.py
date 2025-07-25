# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 18:00:53 2025

@author: Raffaele
"""

import networkx as nx
import pandas as pd

from network_aspects import (assign_initial_sentiments, analyze_central_nodes, compute_mini_cluster_centrality,
                             calculate_sentiment_influence, compute_sentiment_flow_matrix, identify_shared_posts,
                             analyze_post_engagement, get_top_commenters, build_commenter_network, detect_communities_louvain,
                             compute_modularity, calculate_sentiment_influence, analyze_top_influencers)


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
    

def test_identify_shared_posts_finds_common_posts():
    """
    Given a set of comments where two cluster users comment on the same post,
    when identify_shared_posts is called,
    then it should return that post ID and its title.
    """
    #Initialize a comments dataframe with author and post_id columns
    filtered_comments = pd.DataFrame({
        "author": ["user1", "user2", "user3"],
        "post_id": ["post1", "post1", "post2"]
    })
    
    #initialize a posts dataframe with post_id and title columns
    filtered_posts = pd.DataFrame({
        "post_id": ["post1", "post2"],
        "title": ["Shared Theory", "Solo Post"]
    })
    
    #initialize some users being in the cluster
    cluster_nodes = {"user1", "user2"}

    #call the function
    result = identify_shared_posts(filtered_comments, filtered_posts, cluster_nodes)

    #asserts
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert result.iloc[0]["post_id"] == "post1"
    assert result.iloc[0]["title"] == "Shared Theory"
    
def test_identify_shared_posts_only_one_commenter_in_the_cluster():
    """
    Given a dataframe containing a post commented by only one user which is member of a cluster,
    when calling the function identify shared posts,
    then the post should not be included in the result.
    """
    #Initialize a comments dataframe 
    filtered_comments = pd.DataFrame({
        "author":["user1", "user2", "user3"],
        "post_id":["post1", "post2", "post3"] #3 different post ids this time
        })
    #initialize posts dataframe
    filtered_posts = pd.DataFrame({
        "post_id":["post_1", "post_2", "post_3"],
        "title":["A post that is not shared", "Another post not shared", "The third post not shared today"]
        })
     
    #initialize cluster nodes
    cluster_nodes = {"user1", "user3"}
    
    #call the function
    result = identify_shared_posts(filtered_comments, filtered_posts, cluster_nodes)
    
    #asserts
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0

def test_identify_shared_posts_users_comment_multiple_times():
    """
    Given that user1 and user2 commented multiple times on the same post,
    when the function identify shared posts is called,
    then it the post should appear only once in the result.
    """
    #Initialize a comments dataframe
    filtered_comments = pd.DataFrame({
        "author":["user1","user2", "user1", "user2"],
        "post_id":["post1", "post1", "post1", "post1"] #user1 and user2 commented 2 times each on the same post
        })
    #initialize posts dataframe
    filtered_posts = pd.DataFrame({
        "post_id":["post1"],
        "title":["Shared post"]
        })
    #initialize cluster nodes
    cluster_nodes = {"user1", "user2"}
    
    #call the function
    result = identify_shared_posts(filtered_comments, filtered_posts, cluster_nodes)
    
    #asserts
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1

def test_analyze_post_engagement_returns_expected_summary():
    """
    GIVEN a dataset where one post is heavily commented and others are not,
    WHEN calling analyze_post_engagement,
    THEN it should return only that high-engagement post with correct stats.
    """
    #initialize a dataframe with authors and post_ids
    df = pd.DataFrame({
        "author": ["a"] * 5 + ["b", "c", "d", "e", "f"],
        "post_id": ["post1"] * 5 + ["post2"] * 5
    })
    
    #call the function
    result = analyze_post_engagement(df, percentile=50)
    
    #asserts
    assert isinstance(result, pd.DataFrame)
    assert "post_id" in result.columns
    assert "unique_commenters" in result.columns
    assert result.shape[0] == 2
    
    # Make sure both post1 and post2 are present
    post_ids = result["post_id"].tolist()
    assert "post1" in post_ids
    assert "post2" in post_ids

    # Check stats for post2
    post2_data = result[result["post_id"] == "post2"].iloc[0]
    assert post2_data["unique_commenters"] == 5
    assert post2_data["total_comments"] == 5
    assert post2_data["top_commenter_count"] == 1

    # Optionally also check post1
    post1_data = result[result["post_id"] == "post1"].iloc[0]
    assert post1_data["unique_commenters"] == 1
    assert post1_data["total_comments"] == 5
    assert post1_data["top_commenter_count"] == 5
    
def test_analyze_post_engagement_all_posts_have_same_number_of_comments():
    """
    Given that post1, post2 and post3 have all the same number of comments,
    when the function analyze_post_engagement is called,
    then it should return top_posts_ids as empty.
    """
    #initialize a dataframe with authors and post_ids
    df = pd.DataFrame({
        "author":["A"]*3+["B"]*3+["C"]*3, #arrays should have same dimensions; we could even user 9 different authors
        "post_id":["post1"]*3+["post2"]*3+["post3"]*3
        })
    
    #call the function
    result = analyze_post_engagement(df, percentile=90)
    
    #asserts
    assert isinstance(result, pd.DataFrame)
    assert result.shape[0] == 3 #all posts included
    
    #convert result["post_id"] into a list so that we can check that all 3 posts are present
    post_ids = result["post_id"].tolist()
    assert set(post_ids) == {"post1", "post2", "post3"}
    
    # Validate stats for one post (they're all the same)
    for post_id in ["post1", "post2", "post3"]:
        row = result[result["post_id"] == post_id].iloc[0]
        assert row["total_comments"] == 3
        assert row["unique_commenters"] == 1
        assert row["top_commenter_count"] == 3
        
def test_analyze_post_engagement_empty_dataframe():
    """
    Given an empty dataframe of comments,
    when the function is called,
    then it should return an empty dataframe.
    """
    #Initialize an empty dataframe
    df = pd.DataFrame({
        "author":[],
        "post_id":[]
        })
    
    #call the function
    result = analyze_post_engagement(df, percentile=90)
    
    assert result.empty
    
def test_analyze_post_engagement_one_post_only():
    """
    Given a dataframe with one post and few comments, 
    when the function is called,
    then the post should be included in the result regardless of the percentile used.
    """
    #Initialize a dataframe with one post only
    df = pd.DataFrame({
        "author":["author1", "author2", "author3", "author4","author5"],
        "post_id":["post1"]*5 #pandas expect arrays to be the same lenght
        })
    
    #call the function
    result = analyze_post_engagement(df, percentile=90)
    result_99_9 = analyze_post_engagement(df, percentile=99.9)
    
    assert not result.empty
    assert not result_99_9.empty
    

def test_get_top_commenters_returns_most_active_users():
    """
    Given a comment dataframe with several users making different numbers of comments,
    when get_top_commenters is called with top_k=3,
    then it should return the top 3 authors in correct order of activity.
    """
    #Initialize a dataframe with the frequency of comments made by users
    df = pd.DataFrame({
        "author": [
            "user1", "user1", "user1",   # 3 comments
            "user2", "user2",            # 2 comments
            "user3",                     # 1 comment
            "user4", "user4", "user4", "user4"  # 4 comments
        ]
    })
    
    #call the function
    result = get_top_commenters(df, top_k=3)
    
    #asserts
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0] == "user4"  # Most comments
    assert result[1] == "user1"
    assert result[2] == "user2"

def test_get_top_commenters_fewer_than_top_k_commenters():
    """
    Given a dataframe with only 2 authors,
    when calling the function get_top_commenters with k=5,
    then it should return only the top 2 authors.
    """
    #initialie a dataframe with only two authors
    df = pd.DataFrame({
        "author":[
            "user1", "user1",
            "user2", "user2", "user2"
            ]
        })
    
    #call the function
    result = get_top_commenters(df, top_k=5)
    
    assert isinstance(result, list)
    assert len(result) == 2 
    assert result[0] == "user2"
    assert result[1] == "user1"
    
def test_get_top_commenters_empty_dataframe():
    """
    Given an empty dataframe,
    when calling the function get_top_commenters,
    then it should return an empty list.
    """
    #initialize an empty dataframe
    df = pd.DataFrame({
        "author":[]
        })
    #call the function
    result = get_top_commenters(df, top_k=2)
    
    assert result == []
    
def test_get_top_commenters_tie_between_authors():
    """
    Given a dataframe containing two users with the same number of comments,
    when the get_top_comments function is called,
    then it should return both the users are included.
    """
    #initialize a dataframe
    df = pd.DataFrame({
        "author":[
            "user1", "user1",
            "user2", "user2"
            ]
        })
    
    #call the function
    result = get_top_commenters(df, top_k=2)
    
    assert len(result) == 2
    assert result[0] == "user1" #when tied, the alphabetical order is chosen

def test_build_commenter_network_creates_edges_between_shared_post_commenters():
    """
    Given a DataFrame where two top users comment on the same post,
    when build_commenter_network is called,
    then the graph should contain an edge between them with weight 1.
    """
    #initialize a comment dataframe
    comments = pd.DataFrame({
        "author": ["user1", "user2", "user3", "user1", "user3"],
        "post_id": ["postA", "postA", "postB", "postC", "postC"]
    })
    
    #initialize top commenters
    top_commenters = ["user1", "user2", "user3"]
    
    #initialize the graph
    graph = build_commenter_network(comments, top_commenters)
    
    #check that the graph has been created
    assert isinstance(graph, nx.Graph)
    #check that top_commenters are in the graph nodes
    assert set(graph.nodes) == set(top_commenters)
    #check edges between top commenters
    assert ("user1", "user2") in graph.edges
    assert ("user1", "user3") in graph.edges
    #check for weights
    assert graph["user1"]["user2"]["weight"] == 1
    assert graph["user1"]["user3"]["weight"] == 1
    
def test_build_commenter_network_users_comment_on_different_posts():
    """
    Given top commenters who never share the same post,
    when the build_top_commenter function is called,
    then no edges should be created in the graph.
    """
    #initialize a comment dataframe
    comments = pd.DataFrame({
        "author":["user1", "user2", "user3"],
        "post_id":["post_id_not_shared_by_user2_and_user3",
                   "post_id_not_shared_by_user1_and_user3",
                   "post_id_not_shared_by_user2_and_user1"]
        })
    
    #initialize top commenters
    top_commenters = ["user1", "user2", "user3"]
    #initialize the graph
    graph = build_commenter_network(comments, top_commenters)
    #asserts
    assert graph.number_of_edges() == 0
    
def test_build_commenter_network_author_not_in_top_commenter_list():
    """
    Given a comment thread where a non top commenter is present,
    when the function build_commenter_network is called,
    then the user should not appear in the network.
    """
    #initialize a comment dataframe
    comments = pd.DataFrame({
        "author":["user1", "user2", "user3"],
        "post_id":["post_id_1", "post_id_2","post_id_3"]
        })
    #initialize top commenters
    top_commenters = ["user1", "user2"]
    #initialize graph 
    graph = build_commenter_network(comments, top_commenters)
    
    assert set(graph.nodes) == set(top_commenters)
    assert not ("user1", "user3") in graph.edges
    assert not ("user2", "user3") in graph.edges
    
def test_build_commenter_network_multiple_shared_posts_between_same_users():
    """
    Given two users co-commenting on multiple posts,
    when the function build_commenter_network is called,
    then the edge weight between them should reflect the number of posts shared.
    """
    #initialize comments dataframe
    comments = pd.DataFrame({
        "author":["user1", "user2", "user1", "user2"],
        "post_id":["post1", "post1", "post2", "post2"]
        })
    #initialize top commenters
    top_commenters = ["user1", "user2"]
    #initialize graph
    graph = build_commenter_network(comments, top_commenters)
    
    assert graph["user1"]["user2"]["weight"] == 2 #2 is the number of shared posts

def test_detect_communities_louvain_returns_valid_partition():
    """
    Given a graph with two disconnected clusters,
    when detect_communities_louvain is called,
    then it should assign distinct communities to each cluster's nodes.
    """
    #Build graph with two clear communities
    G = nx.Graph()
    G.add_edges_from([("A", "B"), ("B", "C")])         # Cluster 1
    G.add_edges_from([("X", "Y"), ("Y", "Z")])         # Cluster 2
    
    #call the function
    partition = detect_communities_louvain(G)

    assert isinstance(partition, dict)
    assert set(G.nodes()) == set(partition.keys())
    
    communities = set(partition.values())
    assert len(communities) == 2  # Expect 2 communities
    
def test_detect_communities_louvain_graph_with_no_edges():
    """
    Given a graph with nodes but no edges connecting them,
    when the detect_communities_louvain function is called,
    then each node should be in its own community.
    """
    #Build a graph with nodes but no edges
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C"])
    
    #call the function
    partition = detect_communities_louvain(graph)
    communities = set(partition.values())
    
    assert isinstance(partition, dict)
    assert len(communities) == 3 #3 communities expected
    
def test_detect_communities_louvain_fully_connected_graph():
    """
    Given a fully connected graph made of 5 nodes,
    when the detect_communities_louvain function is called,
    then it should return a single community.
    """
    #build the graph
    graph = nx.Graph()
    graph = nx.complete_graph(["A", "B", "C", "D", "E"]) #force a full connected graph
    
    #call the function
    partition = detect_communities_louvain(graph)
    communities = set(partition.values())
    
    assert isinstance(partition, dict)
    assert len(communities) == 1    


def test_compute_modularity_detects_strong_community_structure():
    """
    Given a graph with two dense communities and no interconnection,
    when compute_modularity is called with the correct partition,
    then it should return a high modularity score (close to 1.0).
    """
    G = nx.Graph()

    #Community 0: A-B-C
    G.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

    #Community 1: X-Y-Z
    G.add_edges_from([("X", "Y"), ("Y", "Z"), ("X", "Z")])

    partition = {
        "A": 0, "B": 0, "C": 0,
        "X": 1, "Y": 1, "Z": 1
    }

    modularity = compute_modularity(G, partition)

    assert isinstance(modularity, float)
    assert 0.4 <= modularity <= 1.0  # very strong modular structure
   
def test_compute_modularity_all_nodes_in_one_community():
    """
    Given a graph with multiple edges and nodes placed in the same community,
    when the function compute_modularity is called
    then it should return a low modularity near 0.
    """
    graph = nx.Graph()
    
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D","A")])
    
    partition = {
        "A":0, "B":0, "C":0, "D":0
        }
    
    modularity = compute_modularity(graph, partition)
    
    assert isinstance(modularity, float)
    assert modularity <= 0 #should be smaller than zero or something around zero
    
def test_compute_modularity_each_node_in_its_own_community():
    """
    Given a graph with multiple nodes each in its own community,
    when the function compute_modularity is called,
    then it should return something negative or close to 0.
    """
    graph = nx.Graph()
    
    graph.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D","A")])

    
    partition = {
        "A":0, "B":1, "C":2, "D":3
        }
    
    modularity = compute_modularity(graph, partition)
    
    assert isinstance(modularity, float)
    assert modularity <= 0
    


def test_analyze_top_influencers_returns_expected_output():
    """
    Given a small graph with a clear central node,
    when analyze_top_influencers is called,
    then it should return a subgraph with all nodes
    and top influencers lists for degree, betweenness, and eigenvector centrality.
    """
    graph = nx.Graph()
    graph.add_edges_from([
        ("A", "B"), ("A", "C"), ("A", "D"),  # A is highly connected
        ("D", "E"), ("E", "F")
    ])

    result = analyze_top_influencers(graph, top_n=5)

    #Check structure of result
    assert isinstance(result, dict)
    assert "degree" in result
    assert "betweenness" in result
    assert "eigenvector" in result
    assert "subgraph" in result

    #Check subgraph
    subgraph = result["subgraph"]
    assert isinstance(subgraph, nx.Graph)
    assert subgraph.number_of_nodes() <= graph.number_of_nodes()

    #Check top influencers
    top_degree = result["degree"]
    assert isinstance(top_degree, list)
    assert top_degree[0][0] == "A"  # node A has highest degree

def test_analyze_top_influencers_graph_with_fewer_than_top_n_nodes():
    """
    Given a small graph of 3 nodes,
    when the analyze_top_influencers function is called passing top_n=10,
    then the function should include all nodes.
    """
    #initialize a graph with 3 nodes
    graph = nx.Graph()
    graph.add_edges_from([
        ("A", "B"), ("A", "C")
        ])
    
    #call the function
    result = analyze_top_influencers(graph, top_n=10)
    
    #Check structure of result
    assert isinstance(result, dict)
    assert "degree" in result
    assert "betweenness" in result
    assert "eigenvector" in result
    assert "subgraph" in result
    #Check subgraph
    subgraph = result["subgraph"]
    assert isinstance(subgraph, nx.Graph)
    assert subgraph.number_of_nodes() <= graph.number_of_nodes()

    #Check top influencers
    top_degree = result["degree"]
    assert isinstance(top_degree, list)
    
def test_analyze_top_influencers_graph_with_no_edges():
    """
    Given a graph with isolated nodes and no edges,
    when analyze_top_influencers is called,
    then degree and betweenness scores should be zero,
    and eigenvector scores should all be equal.
    """
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C", "D"])  # 4 isolated nodes

    result = analyze_top_influencers(graph, top_n=5)

    #Subgraph should include all nodes
    subgraph = result["subgraph"]
    assert set(subgraph.nodes) == {"A", "B", "C", "D"}

    #Degree and betweenness scores should be zero
    for node, score in result["degree"]:
        assert score == 0.0

    for node, score in result["betweenness"]:
        assert score == 0.0

    #Eigenvector scores: all equal, each 1/sqrt(4)
    expected_score = 1 / len(subgraph.nodes) ** 0.5
    for node, score in result["eigenvector"]:
        assert score == expected_score

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



