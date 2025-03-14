# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 11:27:09 2025

@author: Raffaele
"""


import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


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
    


def custom_layout(G, scale=3, k=0.15, iterations=50, seed=42):
    """
    Places post_author nodes in a circle, then runs a spring layout
    with those positions fixed, so commenters arrange around them.

    Args:
        G (nx.Graph or nx.DiGraph): Your network (posts + commenters).
        scale (float): Scale for the initial circle layout of posts.
        k (float): Optimal distance between nodes in the spring layout.
        iterations (int): Number of spring layout iterations.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary mapping each node to a 2D position.
    """
    # 1) Convert to undirected if needed
    if isinstance(G, nx.DiGraph):
        G_undirected = G.to_undirected()
    else:
        G_undirected = G

    # 2) Separate out the posts
    posts = [n for n, d in G_undirected.nodes(data=True) if d.get('type') == 'post_author']
    
    # 3) Create a subgraph with just the posts
    post_subgraph = G_undirected.subgraph(posts)
    
    # 4) Place posts in a circle
    pos_posts = nx.circular_layout(post_subgraph, scale=scale, center=(0, 0))

    # 5) Run spring layout for the entire graph, pinning the posts
    #    so they remain in a circle
    pos = nx.spring_layout(
        G_undirected,
        pos=pos_posts,       # start with the circular positions
        fixed=posts,         # don't move the posts
        k=k,
        iterations=iterations,
        seed=seed
    )
    
    return pos


# 1) Build your graph as before
# interaction_graph = build_interaction_graph(posts_df, comments_df) 
# (Make sure it has node attribute 'type' == 'post_author' for posts, 'commenter' for others)

# 2) Generate the layout
pos_dict = custom_layout(interaction_graph, scale=3, k=0.15, iterations=50, seed=42)

# 3) Separate posts and commenters for coloring
posts = [n for n, d in interaction_graph.nodes(data=True) if d.get('type') == 'post_author']
commenters = [n for n, d in interaction_graph.nodes(data=True) if d.get('type') == 'commenter']

# 4) Plot
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(interaction_graph, pos_dict, nodelist=posts, node_color='red', node_size=100, label="Posts")
nx.draw_networkx_nodes(interaction_graph, pos_dict, nodelist=commenters, node_color='blue', node_size=30, label="Commenters")
nx.draw_networkx_edges(interaction_graph, pos_dict, alpha=0.4, edge_color="black")
plt.legend()
plt.title("Donut-Shaped Network Layout")
plt.axis("off")
plt.show()



import numpy as np
from community import community_louvain  # For community detection

# Load data
posts_df = pd.read_csv("../data/onepiece_sentiment_posts_filtered.csv")
df_filtered_comments = pd.read_csv("../data/onepiece_sentiment_comments_filtered.csv")  # Use the filtered comments DataFrame

# Preprocess data
posts_df.dropna(subset=['author_x', 'post_id'], inplace=True)
df_filtered_comments.dropna(subset=['author', 'post_id'], inplace=True)
df_filtered_comments = df_filtered_comments[df_filtered_comments['author'] != '[deleted]']

# Build bipartite graph
bipartite_graph = nx.Graph()

# Add post nodes
for _, post in posts_df.iterrows():
    post_id = post['post_id']
    post_author = post['author_x']
    
    # Add or update post node with 'bipartite' attribute
    bipartite_graph.add_node(post_id, bipartite=0, type='post', author=post_author)

# Add commenter nodes and edges
for _, comment in df_filtered_comments.iterrows():  # Use df_filtered_comments here
    post_id = comment['post_id']
    commenter = comment['author']
    sentiment = comment['sentiment_body']  # Use the sentiment_body column
    
    # Add or update commenter node with 'bipartite' attribute
    bipartite_graph.add_node(commenter, bipartite=1, type='commenter')
    
    # Add edge between post and commenter
    bipartite_graph.add_edge(post_id, commenter, sentiment=sentiment)

# Debug: Print nodes and their attributes
for node, data in bipartite_graph.nodes(data=True):
    print(f"Node: {node}, Attributes: {data}")

# Separate post and commenter nodes
post_nodes = [n for n, d in bipartite_graph.nodes(data=True) if d.get('bipartite') == 0]
commenter_nodes = [n for n, d in bipartite_graph.nodes(data=True) if d.get('bipartite') == 1]

print(f"Number of post nodes: {len(post_nodes)}")
print(f"Number of commenter nodes: {len(commenter_nodes)}")

# Detect communities using Louvain method
partition = community_louvain.best_partition(bipartite_graph)

# Add community information to nodes
for node, community in partition.items():
    bipartite_graph.nodes[node]['community'] = community

# Get unique communities and assign colors
communities = set(partition.values())
print(f"Number of communities detected: {len(communities)}")

# Assign a color to each community
community_colors = {community: plt.cm.tab20(i) for i, community in enumerate(communities)}

# Assign colors to nodes based on their community
node_colors = [community_colors[partition[node]] for node in bipartite_graph.nodes]

# Visualize the bipartite graph with community coloring
pos = nx.spring_layout(bipartite_graph, seed=42)

plt.figure(figsize=(12, 12))

# Draw post nodes (red) with community coloring
nx.draw_networkx_nodes(
    bipartite_graph, pos,
    nodelist=post_nodes,
    node_color=[community_colors[partition[node]] for node in post_nodes],
    node_size=100,
    label="Posts"
)

# Draw commenter nodes (blue) with community coloring
nx.draw_networkx_nodes(
    bipartite_graph, pos,
    nodelist=commenter_nodes,
    node_color=[community_colors[partition[node]] for node in commenter_nodes],
    node_size=30,
    label="Commenters"
)

# Draw edges
nx.draw_networkx_edges(bipartite_graph, pos, alpha=0.4, edge_color="gray")

# Add legend
plt.legend()
plt.title("Bipartite Graph with Communities")
plt.axis("off")
plt.show()

# Get top 10% of nodes by degree centrality
degree_centrality = nx.degree_centrality(bipartite_graph)
top_10_percent = int(len(bipartite_graph.nodes) * 0.1)
top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_10_percent]
top_node_ids = [node for node, _ in top_nodes]

# Assign larger size and different color to top nodes
node_sizes = [100 if node in top_node_ids else 30 for node in bipartite_graph.nodes]
node_colors_top = ['orange' if node in top_node_ids else community_colors[partition[node]] for node in bipartite_graph.nodes]

# Visualize the bipartite graph with highlighted top nodes
plt.figure(figsize=(12, 12))

# Draw all nodes with community coloring
nx.draw_networkx_nodes(
    bipartite_graph, pos,
    node_color=node_colors_top,
    node_size=node_sizes,
)

# Draw edges
nx.draw_networkx_edges(bipartite_graph, pos, alpha=0.4, edge_color="gray")

# Add legend
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Top 10% Nodes'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Commenters'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Posts')
])

plt.title("Bipartite Graph with Highlighted Top Nodes")
plt.axis("off")
plt.show()


# Add to imports
from collections import defaultdict
import random

class SentimentPropagator:
    def __init__(self, graph, resistance_prob=0.1, flip_prob=0.05):
        self.graph = graph
        self.resistance_prob = resistance_prob
        self.flip_prob = flip_prob
        self._initialize_sentiments()
        
    def _initialize_sentiments(self):
        """Initialize sentiment values for all nodes."""
        sentiments = {}
        for node, data in self.graph.nodes(data=True):
            if data.get('type') == 'commenter':
                # Use average sentiment of comments for commenters
                edges = self.graph.edges(node, data=True)
                sentiment = np.mean([d.get('sentiment', 0) for _, _, d in edges])
                sentiments[node] = np.sign(sentiment)  # Convert to -1, 0, 1
            else:
                # Posts get neutral sentiment by default
                sentiments[node] = 0
        nx.set_node_attributes(self.graph, sentiments, 'sentiment')
        self.initial_sentiments = sentiments.copy()
        
    def propagate(self, max_steps=20):
        sentiment_over_time = []
        current_sentiments = self.initial_sentiments.copy()
        
        for step in range(max_steps):
            new_sentiments = {}
            for node in self.graph.nodes():
                if random.random() < self.resistance_prob:
                    new_sentiments[node] = current_sentiments[node]
                    continue
                
                if random.random() < self.flip_prob:
                    new_sentiments[node] = random.choice([-1, 0, 1])
                    continue
                
                neighbors = list(self.graph.neighbors(node))
                if not neighbors:
                    new_sentiments[node] = current_sentiments[node]
                    continue
                
                neighbor_sentiments = [current_sentiments[n] for n in neighbors 
                                     if n in current_sentiments]
                if neighbor_sentiments:
                    new_sentiment = np.mean(neighbor_sentiments)
                    new_sentiments[node] = np.sign(new_sentiment)  # Keep as -1, 0, 1
                else:
                    new_sentiments[node] = current_sentiments[node]
            
            current_sentiments = new_sentiments
            sentiment_over_time.append(Counter(current_sentiments.values()))
            
            if step > 1 and sentiment_over_time[-1] == sentiment_over_time[-2]:
                break
        
        return sentiment_over_time

    def plot_sentiment_evolution(self, sentiment_over_time):
        plt.figure(figsize=(10, 6))
        for sentiment, values in zip([-1, 0, 1], zip(*sentiment_over_time)):
            plt.plot(values, label=['Negative', 'Neutral', 'Positive'][sentiment + 1])
        
        plt.xlabel("Time Step")
        plt.ylabel("Proportion of Sentiments")
        plt.title("Sentiment Propagation Over Time")
        plt.legend()
        plt.grid(True)
        plt.show()


# When building the bipartite graph
for _, comment in df_filtered_comments.iterrows():
    sentiment = comment['sentiment_body']  # Ensure this column exists
    bipartite_graph.add_edge(comment['post_id'], comment['author'], 
                            sentiment=sentiment)

propagator = SentimentPropagator(bipartite_graph)
sentiment_evolution = propagator.propagate()
propagator.plot_sentiment_evolution(sentiment_evolution)