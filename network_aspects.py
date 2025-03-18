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

# Centrality Measures
centrality_measures = {
    'degree': nx.degree_centrality(post_centric_graph),
    'betweenness': nx.betweenness_centrality(post_centric_graph),
    'closeness': nx.closeness_centrality(post_centric_graph)
}

# Convert to DataFrame for easier analysis
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

# Load filtered data
filtered_posts = pd.read_csv("../data/onepiece_sentiment_posts_filtered.csv")  # Adjust path if necessary
filtered_comments = pd.read_csv("../data/onepiece_sentiment_comments_filtered.csv")  # Adjust path if necessary

# Initialize the bipartite graph
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

# Use the existing node positions
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


from community import community_louvain

# Apply the Louvain method for community detection
partition = community_louvain.best_partition(bi_graph_filtered_data, weight='weight')

# Extract degree centrality values
degree_centrality = nx.degree_centrality(bi_graph_filtered_data)
degree_values = list(degree_centrality.values())
# Calculate threshold
threshold = np.percentile(degree_values, 90)

# Filter nodes based on threshold
central_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]

# Create a color map for central nodes
central_node_colors = ["red" if node in central_nodes else "gray" for node in bi_graph_filtered_data.nodes()]
central_node_sizes = [1000 if node in central_nodes else 10 for node in bi_graph_filtered_data.nodes()]

# Generate graph layout
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

# Extract subgraph for central nodes
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
G = nx.Graph()
G.add_nodes_from(present_commenters)
filtered_top_comments = filtered_comments[filtered_comments['author'].isin(present_commenters)]

for post_id in filtered_top_comments['post_id'].unique():
    commenters = filtered_top_comments[filtered_top_comments['post_id'] == post_id]['author'].tolist()
    for i in range(len(commenters)):
        for j in range(i + 1, len(commenters)):
            G.add_edge(commenters[i], commenters[j])

# Visualize the full network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, iterations=20)
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
plt.title("Commenter Network - Top Commenters")
plt.show()

# Analyze the largest connected component (mini-cluster)
largest_cluster = max(nx.connected_components(G), key=len)
mini_cluster_graph = G.subgraph(largest_cluster)

print(f"Number of nodes in the cluster: {mini_cluster_graph.number_of_nodes()}")
print(f"Number of edges in the cluster: {mini_cluster_graph.number_of_edges()}")
print(f"Cluster density: {nx.density(mini_cluster_graph)}")

# Visualize the mini-cluster
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(mini_cluster_graph, seed=42)
nx.draw(mini_cluster_graph, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=10)
plt.title("Mini Cluster of Commenters")
plt.show()

# Centrality analysis
centrality_df = pd.DataFrame({
    'degree': nx.degree_centrality(mini_cluster_graph),
    'betweenness': nx.betweenness_centrality(mini_cluster_graph),
    'closeness': nx.closeness_centrality(mini_cluster_graph)
}).sort_values(by='degree', ascending=False)

print("Centrality measures for the mini-cluster:")
print(centrality_df)

# Identify shared posts in the mini-cluster
shared_posts = filtered_comments[filtered_comments['author'].isin(largest_cluster)].groupby('post_id')['author'].apply(list)
shared_posts = shared_posts[shared_posts.apply(len) > 1]
shared_post_ids = shared_posts.index
shared_post_titles = filtered_posts[filtered_posts['post_id'].isin(shared_post_ids)][['post_id', 'title']]

print("Posts shared by commenters in the mini-cluster:")
print(shared_posts)
print("Titles of shared posts:")
print(shared_post_titles)

# Analyze sentiment distribution for mini-cluster commenters
mini_cluster_comments = filtered_comments[filtered_comments['author'].isin(present_commenters)]
sentiment_counts = mini_cluster_comments['sentiment_category'].value_counts()

plt.figure(figsize=(8, 8))
colors = ['#ff9999', '#66b3ff', '#99ff99']
sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=colors, labels=sentiment_counts.index)
plt.title("Sentiment Distribution for Mini-Cluster Commenters")
plt.ylabel('')
plt.show()


import networkx as nx
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt
import numpy as np

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

# ===== Step 1: Build the Commenter-Commenter Network =====
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

# ===== Step 2: Assign Initial Sentiments =====
def assign_initial_sentiment(row):
    if row > 0:
        return 1  # Positive
    elif row < 0:
        return -1  # Negative
    else:
        return 0  # Neutral

# Aggregate sentiment_body by author and assign initial sentiments
initial_sentiments = filtered_comments.groupby('author')['sentiment_body'].mean().apply(assign_initial_sentiment)
nx.set_node_attributes(commenter_graph, initial_sentiments.to_dict(), name='sentiment')

# Print initial sentiment distribution
initial_distribution = Counter(nx.get_node_attributes(commenter_graph, 'sentiment').values())
print("Initial Sentiment Distribution:")
print(f"Positive: {initial_distribution[1]}")
print(f"Neutral: {initial_distribution[0]}")
print(f"Negative: {initial_distribution[-1]}")

# ===== Step 3: Simulate Sentiment Propagation =====
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
nx.set_node_attributes(commenter_graph, original_sentiments, 'sentiment')

# ===== Step 4: Visualize Sentiment Propagation =====
sentiment_labels = [-1, 0, 1]  # Negative, Neutral, Positive
proportions = {label: [] for label in sentiment_labels}

for distribution in sentiment_evolution:
    total = sum(distribution.values())
    for label in sentiment_labels:
        proportions[label].append(distribution.get(label, 0) / total)

# Plot sentiment evolution
plt.figure(figsize=(10, 6))
for label, values in proportions.items():
    plt.plot(values, label={-1: "Negative", 0: "Neutral", 1: "Positive"}[label])
plt.xlabel("Time Step")
plt.ylabel("Proportion of Sentiments")
plt.title("Sentiment Propagation Over Time")
plt.legend()
plt.grid(True)
plt.show()

# ===== Step 5: Check for Convergence =====
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
