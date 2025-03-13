# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:27:09 2025

@author: Raffaele
"""


import networkx as nx
import pandas as pd


def build_interaction_graph(posts_df, comments_df):
    """
    Constructs a directed graph where:
    - Nodes represent post authors and commenters.
    - Edges represent interactions (commenting on a post).

    Args:
        posts_df (pd.DataFrame): DataFrame containing Reddit posts with 'post_id' and 'author'.
        comments_df (pd.DataFrame): DataFrame containing Reddit comments with 'post_id' and 'author'.

    Returns:
        nx.DiGraph: A directed graph with post-comment interactions.
    """
    G = nx.DiGraph()  # Use a directed graph for interactions

    # Drop NaNs and filter out deleted users
    posts_df = posts_df.dropna(subset=['author', 'post_id'])
    comments_df = comments_df.dropna(subset=['author', 'post_id'])
    comments_df = comments_df[comments_df['author'] != '[deleted]']

    # Add nodes and edges
    for _, post in posts_df.iterrows():
        post_id = post['post_id']
        post_author = post['author']

        # Ensure the post author is added as a node
        G.add_node(post_author, type='post_author')

        # Get all commenters for this post
        post_comments = comments_df[comments_df['post_id'] == post_id]

        for _, comment in post_comments.iterrows():
            commenter = comment['author']
            G.add_node(commenter, type='commenter')

            # Add an edge (post_author -> commenter)
            if G.has_edge(post_author, commenter):
                G[post_author][commenter]['weight'] += 1
            else:
                G.add_edge(post_author, commenter, weight=1)

    return G

# Load the data
posts_df = pd.read_csv("../data/onepiece_posts.csv")
comments_df = pd.read_csv("../data/onepiece_comments.csv")

# Build the graph
interaction_graph = build_interaction_graph(posts_df, comments_df)

# Basic stats
print(f"Number of nodes: {interaction_graph.number_of_nodes()}")
print(f"Number of edges: {interaction_graph.number_of_edges()}")


def compute_network_metrics(G):
    """
    Computes basic network metrics including centralities and degree distribution.

    Args:
        G (nx.Graph or nx.DiGraph): The interaction graph.

    Returns:
        dict: A dictionary containing network statistics.
    """
    # Compute centrality measures
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)

    # Store results in a dictionary
    network_stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "degree_centrality": degree_centrality,
        "betweenness_centrality": betweenness_centrality,
        "closeness_centrality": closeness_centrality
    }

    return network_stats

def get_top_nodes(centrality_dict, top_n=10):
    """
    Retrieves the top N nodes based on a given centrality measure.

    Args:
        centrality_dict (dict): Centrality values for each node.
        top_n (int): Number of top nodes to retrieve.

    Returns:
        list: Top N nodes sorted by centrality.
    """
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

# Compute metrics
network_stats = compute_network_metrics(interaction_graph)

# Display basic network info
print(f"Number of nodes: {network_stats['num_nodes']}")
print(f"Number of edges: {network_stats['num_edges']}")

# Get top 10 nodes by centrality
top_degree = get_top_nodes(network_stats["degree_centrality"])
top_betweenness = get_top_nodes(network_stats["betweenness_centrality"])
top_closeness = get_top_nodes(network_stats["closeness_centrality"])

# Print top influential nodes
print("\nTop 10 Nodes by Degree Centrality:")
for node, value in top_degree:
    print(f"{node}: {value:.4f}")

print("\nTop 10 Nodes by Betweenness Centrality:")
for node, value in top_betweenness:
    print(f"{node}: {value:.4f}")

print("\nTop 10 Nodes by Closeness Centrality:")
for node, value in top_closeness:
    print(f"{node}: {value:.4f}")




# =============================================================================
# # -*- coding: utf-8 -*-
# """
# Script for analyzing and visualizing a bipartite network of Reddit posts and comments.
# """
# 
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# import numpy as np
# 
# # ------------------------------
# # Data Loading and Preprocessing
# # ------------------------------
# 
# def load_and_preprocess_data(posts_path, comments_path):
#     """
#     Load and preprocess the posts and comments data.
# 
#     Args:
#         posts_path (str): Path to the posts CSV file.
#         comments_path (str): Path to the comments CSV file.
# 
#     Returns:
#         tuple: A tuple containing the preprocessed posts and comments DataFrames.
#     """
#     posts_df = pd.read_csv(posts_path)
#     comments_df = pd.read_csv(comments_path)
# 
#     # Preprocess posts data
#     posts_df.dropna(subset=['author', 'created_utc', 'score'], inplace=True)
#     posts_df['selftext'].fillna('', inplace=True)
# 
#     # Preprocess comments data
#     comments_df.dropna(subset=['author', 'created_utc', 'score'], inplace=True)
# 
#     return posts_df, comments_df
# 
# # Load the data
# posts_df, comments_df = load_and_preprocess_data(
#     "../data/onepiece_posts.csv",
#     "../data/onepiece_comments.csv"
# )
# 
# # ------------------------------
# # Graph Construction
# # ------------------------------
# 
# # Build the bipartite graph
# bipartite_graph = nx.Graph()
# 
# for _, post_data in posts_df.iterrows():
#     post_id = post_data['post_id']
#     post_author = post_data['author']
#     
#     # Add post as a node
#     bipartite_graph.add_node(post_id, bipartite=0, author=post_author)
#     
#     # Get commenters for the current post
#     post_comments = comments_df[comments_df['post_id'] == post_id]
#     commenters = post_comments['author']
#     
#     # Add commenters as nodes
#     for commenter in commenters:
#         if commenter != '[deleted]':
#             bipartite_graph.add_node(commenter, bipartite=1)
#             bipartite_graph.add_edge(post_id, commenter)
# 
# # Analyze the bipartite graph
# post_nodes = [node for node, data in bipartite_graph.nodes(data=True) if data['bipartite'] == 0]
# commenter_nodes = [node for node, data in bipartite_graph.nodes(data=True) if data['bipartite'] == 1]
# 
# print(f"Number of post nodes: {len(post_nodes)}")
# print(f"Number of commenter nodes: {len(commenter_nodes)}")
# 
# # ------------------------------
# # Visualization
# # ------------------------------
# 
# def visualize_network(graph, node_color_attr=None, node_size=20, edge_color="gray", title="Network", seed=42):
#     """
#     Visualize a network with optional node coloring.
# 
#     Args:
#         graph (networkx.Graph): The graph to visualize.
#         node_color_attr (str): Node attribute to use for coloring (optional).
#         node_size (int): Size of the nodes.
#         edge_color (str): Color of the edges.
#         title (str): Title of the plot.
#         seed (int): Seed for the layout to ensure reproducibility.
#     """
#     pos = nx.spring_layout(graph, seed=seed)
#     node_colors = None
# 
#     if node_color_attr:
#         node_colors = [data.get(node_color_attr, 0) for _, data in graph.nodes(data=True)]
# 
#     plt.figure(figsize=(10, 8))
#     nx.draw(
#         graph,
#         pos,
#         node_color=node_colors,
#         node_size=node_size,
#         edge_color=edge_color,
#         with_labels=False,
#         cmap=plt.cm.rainbow if node_color_attr else None
#     )
#     plt.title(title)
#     plt.show()
# 
# # Visualize the bipartite graph
# visualize_network(
#     bipartite_graph,
#     node_color_attr='bipartite',
#     node_size=50,
#     title="Bipartite Graph: Posts and Commenters"
# )
# 
# # ------------------------------
# # Centrality Analysis
# # ------------------------------
# 
# # Calculate centrality measures
# degree_centrality = nx.degree_centrality(bipartite_graph)
# betweenness_centrality = nx.betweenness_centrality(bipartite_graph)
# closeness_centrality = nx.closeness_centrality(bipartite_graph)
# 
# # Plot centrality distributions
# plt.figure(figsize=(14, 6))
# 
# # Degree centrality distribution
# plt.subplot(1, 3, 1)
# plt.hist(list(degree_centrality.values()), bins=50, color="blue", alpha=0.7)
# plt.title("Degree Centrality Distribution")
# plt.xlabel("Degree Centrality")
# plt.ylabel("Frequency")
# 
# # Betweenness centrality distribution
# plt.subplot(1, 3, 2)
# plt.hist(list(betweenness_centrality.values()), bins=50, color="green", alpha=0.7)
# plt.title("Betweenness Centrality Distribution")
# plt.xlabel("Betweenness Centrality")
# plt.ylabel("Frequency")
# 
# # Closeness centrality distribution
# plt.subplot(1, 3, 3)
# plt.hist(list(closeness_centrality.values()), bins=50, color="red", alpha=0.7)
# plt.title("Closeness Centrality Distribution")
# plt.xlabel("Closeness Centrality")
# plt.ylabel("Frequency")
# 
# plt.tight_layout()
# plt.show()
# 
# # Print summary statistics
# print("Degree Centrality - Mean:", np.mean(list(degree_centrality.values())))
# print("Betweenness Centrality - Mean:", np.mean(list(betweenness_centrality.values())))
# print("Closeness Centrality - Mean:", np.mean(list(closeness_centrality.values())))
# =============================================================================
