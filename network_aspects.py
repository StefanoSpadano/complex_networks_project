# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:21:07 2025

@author: Raffaele
"""

import pandas as pd
import networkx as nx
import numpy as np

# Load data with error handling
def load_csv(filepath):
    """Load a CSV file into a pandas DataFrame with error handling."""
    try:
        return pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty - {filepath}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Parsing issue in file - {filepath}")
        return None

posts_df = load_csv("../data/onepiece_posts.csv")
comments_df = load_csv("../data/onepiece_comments.csv")

if posts_df is None or comments_df is None:
    raise SystemExit("Data loading failed. Exiting script.")

# Data Preprocessing
def preprocess_dataframe(df, required_columns):
    """Clean and preprocess a DataFrame by removing NaNs from required columns."""
    df.dropna(subset=required_columns, inplace=True)
    if 'selftext' in df.columns:
        df['selftext'].fillna('', inplace=True)  # Fill missing post content

preprocess_dataframe(posts_df, ['author', 'created_utc', 'score'])
preprocess_dataframe(comments_df, ['author', 'created_utc', 'score'])

# Initialize Directed Graph
post_centric_graph = nx.DiGraph()

# Build Graph with comment counts
for _, post_data in posts_df.iterrows():
    post_id = post_data.get('post_id')  # Ensure key exists
    post_author = post_data.get('author')
    
    if not post_id or not post_author:
        continue  # Skip invalid data
    
    post_centric_graph.add_node(post_author, type='post_author', comment_count=0)
    
    post_comments = comments_df[comments_df['post_id'] == post_id]
    
    for commenter in post_comments['author']:
        if commenter == '[deleted]':
            continue  # Skip deleted users
        
        if commenter not in post_centric_graph:
            post_centric_graph.add_node(commenter, type='commenter', comment_count=1)
        else:
            post_centric_graph.nodes[commenter]['comment_count'] += 1
        
        post_centric_graph.add_edge(post_author, commenter, weight=1)

# Compute Basic Graph Metrics
num_nodes = post_centric_graph.number_of_nodes()
num_edges = post_centric_graph.number_of_edges()
print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

# Degree Metrics
degrees = dict(post_centric_graph.degree())
if degrees:
    max_degree_node = max(degrees, key=degrees.get)
    print(f"Node with highest degree: {max_degree_node} ({degrees[max_degree_node]} connections)")

# Centrality Measures first attempt
centrality_measures = {
    'degree': nx.degree_centrality(post_centric_graph),
    'betweenness': nx.betweenness_centrality(post_centric_graph),
    'closeness': nx.closeness_centrality(post_centric_graph)
}

# Convert to DataFrame for easier analysis and print the first 10 to check
centrality_df = pd.DataFrame(centrality_measures)
print(centrality_df.sort_values(by='degree', ascending=False).head(10))
print(centrality_df.sort_values(by='betweenness', ascending=False).head(10))
print(centrality_df.sort_values(by='closeness', ascending=False).head(10))

# Degree Centrality Statistics
degree_centralities = list(centrality_measures['degree'].values())
mean_degree = np.mean(degree_centralities)
median_degree = np.median(degree_centralities)
variance_degree = np.var(degree_centralities)
print(f"Mean: {mean_degree}, Median: {median_degree}, Variance: {variance_degree}")

import matplotlib.pyplot as plt

# Load filtered data, adjust path if necessary
filtered_posts = pd.read_csv("../data/onepiece_sentiment_posts_filtered.csv")  
filtered_comments = pd.read_csv("../data/onepiece_sentiment_comments_filtered.csv")  

# Attempt to visualize the network as a bipartite graph to distinguish the post layer from commenter layer
bi_graph_filtered_data = nx.Graph()

# Build the bipartite graph
for _, post_data in filtered_posts.iterrows():
    post_id = post_data['post_id']
    post_author = post_data['author_x']
    
    bi_graph_filtered_data.add_node(
        post_id,
        bipartite=0,
        author=post_author,
        type='post'
    )
    
    post_comments = filtered_comments[filtered_comments['post_id'] == post_id]
    commenters = post_comments['author']
    
    for commenter in commenters:
        if commenter != '[deleted]':
            bi_graph_filtered_data.add_node(
                commenter,
                bipartite=1,
                type='commenter'
            )
            bi_graph_filtered_data.add_edge(post_id, commenter)

# Save the layout for future reuse
pos = nx.spring_layout(bi_graph_filtered_data)

# Plot refined bipartite graph layout
plt.figure(figsize=(12, 12))
nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=[n for n, d in bi_graph_filtered_data.nodes(data=True) if d['type'] == 'post'], node_color='red', node_size=100, label='Posts')
nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=[n for n, d in bi_graph_filtered_data.nodes(data=True) if d['type'] == 'commenter'], node_color='blue', node_size=50, label='Commenters')
nx.draw_networkx_edges(bi_graph_filtered_data, pos, alpha=0.5)
plt.legend()
plt.title("Refined Bipartite Graph Layout")
plt.show()

# Plot bipartite graph with sentiment-based edge coloring
plt.figure(figsize=(15, 15))
edge_colors = []
edge_opacities = []

for u, v in bi_graph_filtered_data.edges():
    sentiment = filtered_comments.loc[
        (filtered_comments['post_id'] == u) & (filtered_comments['author'] == v),
        'sentiment_body'
    ]
    
    if sentiment.empty:
        continue

    sentiment_value = sentiment.values[0]
    if sentiment_value > 0:
        edge_colors.append('green')
    elif sentiment_value < 0:
        edge_colors.append('red')
    else:
        edge_colors.append('gray')
    
    edge_opacities.append(min(1, max(0.1, abs(sentiment_value))))

nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=[n for n, d in bi_graph_filtered_data.nodes(data=True) if d['type'] == 'post'], node_color='red', node_size=100, label='Posts')
nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=[n for n, d in bi_graph_filtered_data.nodes(data=True) if d['type'] == 'commenter'], node_color='blue', node_size=50, label='Commenters')

for (u, v), color, opacity in zip(bi_graph_filtered_data.edges(), edge_colors, edge_opacities):
    nx.draw_networkx_edges(
        bi_graph_filtered_data,
        pos,
        edgelist=[(u, v)],
        edge_color=color,
        alpha=opacity,
        width=1
    )

plt.legend()
plt.title("Bipartite Graph with Sentiment Weights (Colored Edges)")
plt.show()


#Attempt at community detection algorithm

from community import community_louvain

# Apply the Louvain method for community detection
partition = community_louvain.best_partition(bi_graph_filtered_data, weight='weight')

# Extract degree centrality values
degree_centrality = nx.degree_centrality(bi_graph_filtered_data)
degree_values = list(degree_centrality.values())
# The value for the threshold was chosen to exclude most of the nodes and include only those with higher degree
threshold = np.percentile(degree_values, 90)

# Filter nodes based on threshold
central_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]

# Create a color map for central nodes to distinguish them from the other nodes
central_node_colors = ["red" if node in central_nodes else "gray" for node in bi_graph_filtered_data.nodes()]
central_node_sizes = [1000 if node in central_nodes else 10 for node in bi_graph_filtered_data.nodes()]

# Use before defined graph layout
pos = nx.spring_layout(bi_graph_filtered_data)

# Plot the graph highlighting central nodes
plt.figure(figsize=(15, 15))
nx.draw(
    bi_graph_filtered_data,
    pos,
    node_color=central_node_colors,
    node_size=central_node_sizes,
    edge_color='gray',
    alpha=0.7,
    with_labels=False
)
plt.title("Graph Highlighting Central Nodes (90th Percentile Threshold)")
plt.show()

# Extract subgraph for central nodes to show only those belonging to the central zone
central_subgraph = bi_graph_filtered_data.subgraph(central_nodes)
central_node_sizes_subgraph = [100 if node in central_nodes else 10 for node in central_subgraph.nodes()]
subgraph_community_colors = [partition[node] for node in central_subgraph.nodes()]

# Plot the subgraph with community detection applied
plt.figure(figsize=(12, 12))
nx.draw(
    central_subgraph,
    pos,
    node_color=subgraph_community_colors,
    node_size=central_node_sizes_subgraph,
    alpha=0.7,
    with_labels=False,
    edge_color='gray'
)
plt.title("Zoomed-In View: Central Nodes and Their Communities")
plt.show()

# Compute betweenness and closeness centrality for central nodes
betweenness_centrality = nx.betweenness_centrality(bi_graph_filtered_data)
central_betweenness = {node: betweenness_centrality[node] for node in central_nodes}

closeness_centrality = nx.closeness_centrality(bi_graph_filtered_data)
central_closeness = {node: closeness_centrality[node] for node in central_nodes}

# Plot betweenness and closeness centrality distributions
plt.figure(figsize=(14, 6))

# Betweenness Centrality Plot
plt.subplot(1, 2, 1)
plt.hist(list(central_betweenness.values()), bins=30, color='skyblue', edgecolor='black')
plt.title('Betweenness Centrality of Central Nodes')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Frequency')

# Closeness Centrality Plot
plt.subplot(1, 2, 2)
plt.hist(list(central_closeness.values()), bins=30, color='salmon', edgecolor='black')
plt.title('Closeness Centrality of Central Nodes')
plt.xlabel('Closeness Centrality')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Compute degree centrality
degree_centrality = nx.degree_centrality(bi_graph_filtered_data)
degree_values = list(degree_centrality.values())
threshold = np.percentile(degree_values, 90)
central_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]

# Compute comment counts per post and per commenter
post_comment_counts = {}
commenter_comment_counts = {}

for node in bi_graph_filtered_data.nodes():
    if bi_graph_filtered_data.nodes[node]["bipartite"] == 0:  # Assuming 0 represents posts
        post_comment_counts[node] = bi_graph_filtered_data.degree(node)
    else:  # Assuming 1 represents commenters
        commenter_comment_counts[node] = bi_graph_filtered_data.degree(node)

# Extract high centrality posts and commenters
high_centrality_posts = [post_comment_counts[node] for node in central_nodes if node in post_comment_counts]
high_centrality_commenters = [commenter_comment_counts[node] for node in central_nodes if node in commenter_comment_counts]

# Plot distributions
plt.figure(figsize=(14, 6))

# Plot distribution of comments on high centrality posts
plt.subplot(1, 2, 1)
plt.hist(high_centrality_posts, bins=30, color='blue', edgecolor='black')
plt.title('Distribution of Comments on High Centrality Posts')
plt.xlabel('Number of Comments')
plt.ylabel('Frequency')

# Plot distribution of comments by high centrality commenters
plt.subplot(1, 2, 2)
plt.hist(high_centrality_commenters, bins=30, color='green', edgecolor='black')
plt.title('Distribution of Comments by High Centrality Commenters')
plt.xlabel('Number of Comments')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Load data
filtered_posts = pd.read_csv("../data/onepiece_sentiment_posts_filtered.csv")
filtered_comments = pd.read_csv("../data/onepiece_sentiment_comments_filtered.csv")

# Create pivot table for comments count by author and post
heatmap_data = filtered_comments.pivot_table(index='author', columns='post_id', aggfunc='size', fill_value=0)

# Calculate post engagement and identify high engagement posts
#Only those falling inside the top 10% were labeled as the high_engagement_posts
post_engagement = heatmap_data.sum(axis=0)
threshold = post_engagement.quantile(0.9)  # Top 10% threshold
high_engagement_posts = post_engagement[post_engagement > threshold].index

# Analyze commenters for high engagement posts
high_engagement_comments = filtered_comments[filtered_comments['post_id'].isin(high_engagement_posts)]
commenter_analysis = high_engagement_comments.groupby('post_id').agg(
    unique_commenters=('author', 'nunique'),
    total_comments=('author', 'count'),
    top_commenter=('author', lambda x: x.value_counts().idxmax()),
    top_commenter_count=('author', lambda x: x.value_counts().max())
).reset_index()

# Clean and check top commenters
filtered_comments['author_clean'] = filtered_comments['author'].str.strip().str.lower()
top_commenters_list = ['vinsmokewhoswho', 'kidelaleron', 'totally_not_a_reply', 'kerriazes',
                       'idkdidkkdkdj', 'scaptastic', 'nicentra', 'hinrik96']
top_commenters_clean = [x.lower() for x in top_commenters_list]
missing_commenters = [commenter for commenter in top_commenters_clean if commenter not in filtered_comments['author_clean'].values]
present_commenters = [commenter for commenter in top_commenters_clean if commenter in filtered_comments['author_clean'].values]

# Build commenter network
commenter_network = nx.Graph()
commenter_network.add_nodes_from(present_commenters)
filtered_top_comments = filtered_comments[filtered_comments['author'].isin(present_commenters)]

for post_id in filtered_top_comments['post_id'].unique():
    commenters = filtered_top_comments[filtered_top_comments['post_id'] == post_id]['author'].tolist()
    for i in range(len(commenters)):
        for j in range(i + 1, len(commenters)):
            commenter_network.add_edge(commenters[i], commenters[j])

# Visualize the full network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(commenter_network, k=0.15, iterations=20)
nx.draw_networkx_nodes(commenter_network, pos, node_size=500, node_color='skyblue')
nx.draw_networkx_edges(commenter_network, pos, width=2, alpha=0.5, edge_color='gray')
nx.draw_networkx_labels(commenter_network, pos, font_size=12, font_weight='bold')
plt.title("Commenter Network - Top Commenters")
plt.show()

# Analyze the largest connected component (mini-cluster)
largest_cluster = max(nx.connected_components(commenter_network), key=len)
mini_cluster_graph = commenter_network.subgraph(largest_cluster)

print(f"Number of nodes in the cluster: {mini_cluster_graph.number_of_nodes()}")
print(f"Number of edges in the cluster: {mini_cluster_graph.number_of_edges()}")
print(f"Cluster density: {nx.density(mini_cluster_graph)}")

# Visualize the mini-cluster of commenters
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(mini_cluster_graph, seed=42)
nx.draw(mini_cluster_graph, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=10)
plt.title("Mini Cluster of Commenters")
plt.show()

# Centrality analysis applied to the mini cluster of commenters just found; saving them in a dataframe
centrality_df_mini_cluster = pd.DataFrame({
    'degree': nx.degree_centrality(mini_cluster_graph),
    'betweenness': nx.betweenness_centrality(mini_cluster_graph),
    'closeness': nx.closeness_centrality(mini_cluster_graph)
}).sort_values(by='degree', ascending=False)

print("Centrality measures for the mini-cluster:")
print(centrality_df_mini_cluster)

# Identify shared posts in the mini-cluster
shared_posts = filtered_comments[filtered_comments['author'].isin(largest_cluster)].groupby('post_id')['author'].apply(list)
shared_posts = shared_posts[shared_posts.apply(len) > 1]
shared_post_ids = shared_posts.index
shared_post_titles = filtered_posts[filtered_posts['post_id'].isin(shared_post_ids)][['post_id', 'title']]

print("Posts shared by commenters in the mini-cluster:")
print(shared_posts)
print("Titles of shared posts:")
print(shared_post_titles)

# Analyze sentiment distribution for mini-cluster commenters and visualize it as a pie chart
mini_cluster_comments = filtered_comments[filtered_comments['author'].isin(present_commenters)]
sentiment_counts = mini_cluster_comments['sentiment_category'].value_counts()

plt.figure(figsize=(8, 8))
colors = ['#ff9999', '#66b3ff', '#99ff99']
sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=colors, labels=sentiment_counts.index)
plt.title("Sentiment Distribution for Mini-Cluster Commenters")
plt.ylabel('')
plt.show()


#Follows an attempt at simulating propagation of sentiments across the network of commenters
#who commented on the same post

from collections import Counter
import random


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# Build the Commenter-Commenter Network 
commenter_graph = nx.Graph()

# Create edges between commenters who commented on the same post
for post_id, post_comments in filtered_comments.groupby('post_id'):
    commenters = post_comments['author'].unique()
    for i in range(len(commenters)):
        for j in range(i + 1, len(commenters)):
            commenter1, commenter2 = commenters[i], commenters[j]
            if not commenter_graph.has_edge(commenter1, commenter2):
                commenter_graph.add_edge(commenter1, commenter2, weight=0)
            commenter_graph[commenter1][commenter2]['weight'] += 1

# Assign sentiments based on sentiment_body saved earlier

def assign_initial_sentiment(row):
    if row > 0:
        return 1  # Positive
    elif row < 0:
        return -1  # Negative
    else:
        return 0  # Neutral

# Aggregate sentiment_body by author and assign initial sentiments
# here I used average sentiments for each comments authored by the same commenter

initial_sentiments = filtered_comments.groupby('author')['sentiment_body'].mean().apply(assign_initial_sentiment)
nx.set_node_attributes(commenter_graph, initial_sentiments.to_dict(), name='sentiment')

# Print initial sentiment distribution

initial_distribution = Counter(nx.get_node_attributes(commenter_graph, 'sentiment').values())
print("Initial Sentiment Distribution:")
print(f"Positive: {initial_distribution[1]}")
print(f"Neutral: {initial_distribution[0]}")
print(f"Negative: {initial_distribution[-1]}")

# Simulate sentiment propagation 

def propagate_sentiments(graph, max_steps=10, resistance_prob=0.05, flip_prob=0.1):
    sentiment_over_time = []
    current_sentiments = nx.get_node_attributes(graph, 'sentiment')
    sentiment_over_time.append(Counter(current_sentiments.values()))

    for step in range(max_steps):
        new_sentiments = {}
        for node in graph.nodes:
            if random.random() < resistance_prob:
                new_sentiments[node] = graph.nodes[node]['sentiment']
                continue

            if random.random() < flip_prob:
                new_sentiments[node] = random.choice([-1, 0, 1])
                continue

            # Weighted influence from neighbors
            neighbors = list(graph.neighbors(node))
            weighted_sum = sum(
                graph.nodes[neighbor]['sentiment'] * graph[node][neighbor]['weight']
                for neighbor in neighbors
            )
            new_sentiments[node] = 1 if weighted_sum > 0 else -1 if weighted_sum < 0 else 0

        nx.set_node_attributes(graph, new_sentiments, 'sentiment')
        sentiment_distribution = Counter(new_sentiments.values())
        sentiment_over_time.append(sentiment_distribution)

        # Check for convergence
        if sentiment_over_time[-1] == sentiment_over_time[-2]:
            print(f"Converged at step {step + 1}")
            break

    return sentiment_over_time

# Backup original sentiments

original_sentiments = nx.get_node_attributes(commenter_graph, 'sentiment')

# Run sentiment propagation
sentiment_evolution = propagate_sentiments(commenter_graph, max_steps=50)

# Restore original sentiments
# was used during testing to check for possible different results obtained from the sentiment propagation

nx.set_node_attributes(commenter_graph, original_sentiments, 'sentiment')

# Visualize sentiment propagation 
sentiment_labels = [-1, 0, 1]  # Negative, Neutral, Positive
proportions = {label: [] for label in sentiment_labels}

for distribution in sentiment_evolution:
    total = sum(distribution.values())
    for label in sentiment_labels:
        proportions[label].append(distribution.get(label, 0) / total)

# Plot sentiment evolution over a fixed number of times steps

plt.figure(figsize=(10, 6))
for label, values in proportions.items():
    plt.plot(values, label={-1: "Negative", 0: "Neutral", 1: "Positive"}[label])
plt.xlabel("Time Step")
plt.ylabel("Proportion of Sentiments")
plt.title("Sentiment Propagation Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Check for Convergence 
# during the sentiment propagation attempt all the parameters were chosen in heuristic way

convergence_threshold = 0.001
converged = True

for t in range(1, len(sentiment_evolution)):
    prev_dist = sentiment_evolution[t - 1]
    curr_dist = sentiment_evolution[t]
    total_nodes = sum(prev_dist.values())

    diffs = [
        abs(curr_dist.get(label, 0) / total_nodes - prev_dist.get(label, 0) / total_nodes)
        for label in [-1, 0, 1]
    ]

    if any(diff > convergence_threshold for diff in diffs):
        converged = False
        break

print("The system has reached convergence." if converged else "The system has NOT converged yet.")


import copy


# ===== Step 1: Shift Edge Weights for Community Detection =====
def shift_edge_weights(graph):
    """
    Shift edge weights to make them non-negative for community detection.
    
    Args:
        graph (nx.Graph): The input graph with potentially negative edge weights.
    
    Returns:
        nx.Graph: A copy of the graph with shifted edge weights.
    """
    shifted_graph = copy.deepcopy(graph)
    min_weight = min(d['weight'] for _, _, d in shifted_graph.edges(data=True))
    for u, v, d in shifted_graph.edges(data=True):
        d['weight'] += abs(min_weight)
    return shifted_graph

# Shift edge weights and perform community detection
shifted_graph = shift_edge_weights(commenter_graph)
partition = community_louvain.best_partition(shifted_graph)

# Analyze and print community sizes
community_sizes = Counter(partition.values())
print("Community Sizes (number of nodes per community):")
for community, size in community_sizes.items():
    print(f"Community {community}: {size} nodes")

# Assign communities as node attributes for further analysis
nx.set_node_attributes(commenter_graph, partition, name='community')

# ===== Step 2: Analyze Sentiment Distribution per Community =====
def analyze_community_sentiments(graph, partition):
    """
    Analyze sentiment distribution for each community.
    
    Args:
        graph (nx.Graph): The graph with node sentiments.
        partition (dict): A dictionary mapping nodes to their communities.
    
    Returns:
        dict: A dictionary mapping communities to their sentiment distributions.
    """
    community_sentiments = {c: Counter() for c in set(partition.values())}
    for node, community in partition.items():
        sentiment = graph.nodes[node]['sentiment']
        community_sentiments[community][sentiment] += 1
    return community_sentiments

# Populate and print sentiment distribution per community
community_sentiments = analyze_community_sentiments(commenter_graph, partition)
print("Sentiment Distribution per Community:")
for community, sentiments in community_sentiments.items():
    total = sum(sentiments.values())
    print(f"Community {community}:")
    for sentiment, count in sentiments.items():
        proportion = count / total
        label = {1: "Positive", 0: "Neutral", -1: "Negative"}[sentiment]
        print(f"  {label}: {count} ({proportion:.2%})")

# ===== Step 3: Visualize Communities =====
def visualize_communities(graph, partition, highlight_nodes=None, title="Commenter Network"):
    """
    Visualize the graph with nodes colored by their communities.
    Optionally highlight specific nodes (e.g., top influencers).
    
    Args:
        graph (nx.Graph): The graph to visualize.
        partition (dict): A dictionary mapping nodes to their communities.
        highlight_nodes (set): A set of nodes to highlight (e.g., top influencers).
        title (str): Title of the plot.
    """
    # Assign colors based on communities
    num_communities = len(set(partition.values()))
    color_map = plt.cm.rainbow
    colors = [color_map(i / num_communities) for i in range(num_communities)]
    community_colors = {community: colors[i] for i, community in enumerate(set(partition.values()))}
    node_colors = [community_colors[partition[node]] for node in graph.nodes]

    # Assign node sizes based on highlighting
    node_sizes = [100 if node in highlight_nodes else 30 for node in graph.nodes] if highlight_nodes else 30

    # Create the figure and axes
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, seed=42)  # Use spring_layout for consistency

    # Draw the graph
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=False,
        node_size=node_sizes,
        node_color=node_colors,
        edge_color='gray',
        alpha=0.7
    )

    plt.title(title, fontsize=16)
    plt.show()

# Visualize the full graph with communities
visualize_communities(commenter_graph, partition, title="Commenter Network Colored by Communities")

# ===== Step 4: Analyze Top Influencers (Optimized) =====
def analyze_top_influencers(graph, top_n=500):
    """
    Analyze top influencers in the graph using centrality measures.
    First, select the top `top_n` nodes by degree centrality.
    Then, calculate betweenness and eigenvector centrality only for this subset.
    
    Args:
        graph (nx.Graph): The graph to analyze.
        top_n (int): The number of top nodes to consider.
    
    Returns:
        dict: A dictionary containing top influencers by different centrality measures.
    """
    # Step 1: Select top nodes by degree centrality
    degree_centrality = nx.degree_centrality(graph)
    top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_node_ids = [node for node, _ in top_nodes]

    # Create a subgraph from the top nodes
    subgraph = graph.subgraph(top_node_ids)
    print(f"Top Commenters Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

    # Step 2: Calculate betweenness centrality for the subgraph
    betweenness_centrality = nx.betweenness_centrality(subgraph, normalized=True)
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    # Step 3: Calculate eigenvector centrality for the subgraph
    eigenvector_centrality = nx.eigenvector_centrality(subgraph)
    top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    # Step 4: Get top degree centrality nodes (already computed)
    top_degree = top_nodes[:5]

    print("Top Influencers by Degree Centrality:", top_degree)
    print("Top Influencers by Betweenness Centrality:", top_betweenness)
    print("Top Influencers by Eigenvector Centrality:", top_eigenvector)

    return {
        "degree": top_degree,
        "betweenness": top_betweenness,
        "eigenvector": top_eigenvector,
        "subgraph": subgraph
    }

# Analyze top influencers
influencers = analyze_top_influencers(commenter_graph, top_n=500)  # Limit to top 500 nodes

# ===== Step 5: Calculate Sentiment Influence Scores =====
def calculate_sentiment_influence(subgraph):
    """
    Calculate sentiment influence scores for nodes in the subgraph.
    
    Args:
        subgraph (nx.Graph): The subgraph to analyze.
    
    Returns:
        dict: A dictionary mapping nodes to their sentiment influence scores.
    """
    influence_scores = {node: 0 for node in subgraph.nodes}
    for node in subgraph.nodes:
        for neighbor in subgraph.neighbors(node):
            if subgraph.nodes[node]['sentiment'] == subgraph.nodes[neighbor]['sentiment']:
                influence_scores[node] += 1
    return influence_scores

# Get top sentiment influencers
influence_scores = calculate_sentiment_influence(influencers["subgraph"])
top_sentiment_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top Influencers by Sentiment Influence:", top_sentiment_influencers)

# ===== Step 6: Visualize Subgraph with Top Nodes Highlighted =====
top_nodes = {node for node, _ in influencers["degree"] + influencers["betweenness"] + influencers["eigenvector"] + top_sentiment_influencers}
visualize_communities(influencers["subgraph"], partition, highlight_nodes=top_nodes, title="Top Nodes Highlighted by Community")


# ===== Step 1: Compute Modularity Score =====
def compute_modularity(graph, partition):
    """
    Compute the modularity score for the given graph and partition.
    
    Args:
        graph (nx.Graph): The input graph.
        partition (dict): A dictionary mapping nodes to their communities.
    
    Returns:
        float: The modularity score.
    """
    modularity = community_louvain.modularity(partition, graph)
    print(f"Modularity Score: {modularity}")
    return modularity

# Compute modularity score
modularity = compute_modularity(commenter_graph, partition)

# ===== Step 2: Compute Sentiment Flow Matrix =====
def compute_sentiment_flow_matrix(graph, partition):
    """
    Compute the sentiment flow matrix between communities.
    
    Args:
        graph (nx.Graph): The input graph.
        partition (dict): A dictionary mapping nodes to their communities.
    
    Returns:
        np.ndarray: A matrix where each cell (i, j) represents the sentiment flow from community i to j.
    """
    num_communities = len(set(partition.values()))
    sentiment_flow_matrix = np.zeros((num_communities, num_communities))

    # Iterate over edges in the graph
    for u, v, d in graph.edges(data=True):
        community_u = partition[u]
        community_v = partition[v]
        weight = d.get('weight', 1)  # Default weight to 1 if not set
        sentiment_u = graph.nodes[u]['sentiment']
        sentiment_v = graph.nodes[v]['sentiment']

        # Increment the flow based on sentiment alignment
        if sentiment_u == sentiment_v:
            sentiment_flow_matrix[community_u, community_v] += weight

    return sentiment_flow_matrix

# Compute sentiment flow matrix
sentiment_flow_matrix = compute_sentiment_flow_matrix(commenter_graph, partition)

# Print the sentiment flow matrix
print("Sentiment Flow Matrix:")
print(sentiment_flow_matrix)

import seaborn as sns

# ===== Step 3: Visualize Sentiment Flow Matrix =====
def visualize_sentiment_flow(matrix):
    """
    Visualize the sentiment flow matrix as a heatmap.
    
    Args:
        matrix (np.ndarray): The sentiment flow matrix.
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title("Sentiment Flow Matrix Heatmap")
    plt.xlabel("Community")
    plt.ylabel("Community")
    plt.show()

# Visualize the sentiment flow matrix
visualize_sentiment_flow(sentiment_flow_matrix)

# ===== Step 4: Compute Inter and Intra Flow Values =====
def compute_flow_values(matrix):
    """
    Compute inter-community and intra-community flow values.
    
    Args:
        matrix (np.ndarray): The sentiment flow matrix.
    
    Returns:
        tuple: A tuple containing two lists:
            - inter_flows: List of (i, j, flow) for inter-community flows.
            - intra_flows: List of (i, flow) for intra-community flows.
    """
    num_communities = matrix.shape[0]
    inter_flows = []
    intra_flows = []

    for i in range(num_communities):
        for j in range(num_communities):
            if i == j:
                # Intra-community flow
                intra_flows.append((i, matrix[i, j]))
            else:
                # Inter-community flow
                if matrix[i, j] > 0:
                    inter_flows.append((i, j, matrix[i, j]))

    # Sort flows by value in descending order
    inter_flows = sorted(inter_flows, key=lambda x: x[2], reverse=True)
    intra_flows = sorted(intra_flows, key=lambda x: x[1], reverse=True)

    return inter_flows, intra_flows

# Compute inter and intra flow values
inter_flows, intra_flows = compute_flow_values(sentiment_flow_matrix)

# Print top inter-community flows
print("Top Inter-Community Flows:")
for i, j, flow in inter_flows[:10]:  # Top 10 flows
    print(f"Community {i} ↔ Community {j}: {flow:.2f}")

# Print top intra-community flows
print("Top Intra-Community Flows:")
for i, flow in intra_flows[:10]:  # Top 10 flows
    print(f"Community {i}: {flow:.2f}")