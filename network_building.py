# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 12:55:33 2024

@author: Raffaele
"""
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def load_and_preprocess_data(posts_path, comments_path):
    """
    Load and preprocess the posts and comments data.

    Args:
        posts_path (str): Path to the posts CSV file.
        comments_path (str): Path to the comments CSV file.

    Returns:
        tuple: A tuple containing the preprocessed posts and comments DataFrames.
    """
    try:
        # Load data
        posts_df = pd.read_csv(posts_path)
        comments_df = pd.read_csv(comments_path)

        # Preprocess posts data
        posts_df.dropna(subset=['author', 'created_utc', 'score'], inplace=True)
        posts_df['selftext'].fillna('', inplace=True)

        # Preprocess comments data
        comments_df.dropna(subset=['author', 'created_utc', 'score'], inplace=True)

        return posts_df, comments_df

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None
    except KeyError as e:
        print(f"Missing required column: {e}")
        return None, None

# Example usage
posts_df, comments_df = load_and_preprocess_data(
    "../data/onepiece_posts.csv",
    "../data/onepiece_comments.csv"
)
if posts_df is not None and comments_df is not None:
    print("Data loaded and preprocessed successfully!")

# Step 1: Initialize the graph
post_centric_graph = nx.DiGraph()  # Directed graph for post-comment relationships

# Step 2: Build the graph with comment counts
for _, post_data in posts_df.iterrows():  # Iterate over each post in posts_df
    post_id = post_data['post_id']  # Assuming 'post_id' is the unique identifier for posts
    post_author = post_data['author']
    
    # Add the post author as a node
    post_centric_graph.add_node(post_author, type='post_author', comment_count=0)  # Initialize comment count
    
    # Get commenters for the current post
    post_comments = comments_df[comments_df['post_id'] == post_id]
    commenters = post_comments['author']
    
    # Add edges between post author and commenters, and track comment counts
    for commenter in commenters:
        if commenter != '[deleted]':  # Skip deleted users
            if commenter not in post_centric_graph:  # If the commenter node doesn't exist, add it
                post_centric_graph.add_node(commenter, type='commenter', comment_count=1)  # Initialize comment count for commenter
            else:  # If the commenter node already exists, increment their comment count
                post_centric_graph.nodes[commenter]['comment_count'] += 1
                
            post_centric_graph.add_edge(post_author, commenter, weight=1)  # Add edge with weight

# Step 3: Compute basic metrics
num_nodes = post_centric_graph.number_of_nodes()
num_edges = post_centric_graph.number_of_edges()

print(f"Number of nodes: {num_nodes}")
print(f"Number of edges: {num_edges}")

# Degree distribution
degrees = dict(post_centric_graph.degree())
max_degree_node = max(degrees, key=degrees.get)
print(f"Node with highest degree: {max_degree_node} ({degrees[max_degree_node]} connections)")

# Step 4: Visualize the network
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(post_centric_graph, seed=42)  # Layout for visualization
nx.draw(post_centric_graph, pos, with_labels=False, node_size=20, edge_color="gray")
plt.title("Post-Centric Network")
plt.show()


# Calculate Degree Centrality
degree_centrality = nx.degree_centrality(post_centric_graph)


# Calculate Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(post_centric_graph)

# Calculate Closeness Centrality
closeness_centrality = nx.closeness_centrality(post_centric_graph)

# Optionally, you can convert centralities into DataFrames for easy analysis

centrality_df = pd.DataFrame({
    'degree': degree_centrality,
    'betweenness': betweenness_centrality,
    'closeness': closeness_centrality
})

# Show the top 10 highest centrality nodes for each type
print(centrality_df.sort_values(by='degree', ascending=False).head(10))
print(centrality_df.sort_values(by='betweenness', ascending=False).head(10))
print(centrality_df.sort_values(by='closeness', ascending=False).head(10))

degree_centralities = list(nx.degree_centrality(post_centric_graph).values())

plt.figure(figsize=(12, 6))
plt.hist(degree_centralities, bins=50, color="blue", alpha=0.7)
plt.xlabel("Degree Centrality")
plt.ylabel("Frequency")
plt.title("Degree Centrality Distribution (Linear Scale)")
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
plt.hist(degree_centralities, bins=50, color="green", alpha=0.7, log=True)
plt.xlabel("Degree Centrality")
plt.ylabel("Log(Frequency)")
plt.title("Degree Centrality Distribution (Log Y-Axis)")
plt.grid(True)
plt.show()

mean_degree = np.mean(degree_centralities)
median_degree = np.median(degree_centralities)
variance_degree = np.var(degree_centralities)
print(f"Mean: {mean_degree}, Median: {median_degree}, Variance: {variance_degree}")


#better binning for the log distribution under here
# =============================================================================
# # Define finer bins for degree centrality
# bins = np.linspace(0, 0.1, 50)  # 50 bins between 0 and 0.1
# 
# # Plot the histogram
# plt.hist(degree_centralities, bins=bins, edgecolor='k', log=True)
# plt.xlabel('Degree Centrality')
# plt.ylabel('Frequency (log scale)')
# plt.title('Degree Centrality Distribution')
# plt.grid(True, which="both", linestyle='--', linewidth=0.5)
# plt.show()
# 
# =============================================================================


# Compute betweenness centrality
betweenness_centrality = nx.betweenness_centrality(post_centric_graph)

# Compute closeness centrality
closeness_centrality = nx.closeness_centrality(post_centric_graph)

# Convert centrality values to lists (for later plotting)
betweenness_values = list(betweenness_centrality.values())
closeness_values = list(closeness_centrality.values())

# Plotting distributions
plt.figure(figsize=(14, 6))

# Betweenness centrality distribution (log-log scale)
plt.subplot(1, 2, 1)
plt.hist(betweenness_values, bins=50, density=True, log=True)
plt.title('Betweenness Centrality Distribution (Log Scale)')
plt.xlabel('Betweenness Centrality')
plt.ylabel('Frequency (log scale)')

# Closeness centrality distribution (log-log scale)
plt.subplot(1, 2, 2)
plt.hist(closeness_values, bins=50, density=True, log=True)
plt.title('Closeness Centrality Distribution (Log Scale)')
plt.xlabel('Closeness Centrality')
plt.ylabel('Frequency (log scale)')

plt.tight_layout()
plt.show()

# Summary statistics
betweenness_mean = np.mean(betweenness_values)
betweenness_median = np.median(betweenness_values)
betweenness_variance = np.var(betweenness_values)

closeness_mean = np.mean(closeness_values)
closeness_median = np.median(closeness_values)
closeness_variance = np.var(closeness_values)

print("Betweenness Centrality - Mean: ", betweenness_mean)
print("Betweenness Centrality - Median: ", betweenness_median)
print("Betweenness Centrality - Variance: ", betweenness_variance)

print("\nCloseness Centrality - Mean: ", closeness_mean)
print("Closeness Centrality - Median: ", closeness_median)
print("Closeness Centrality - Variance: ", closeness_variance)


# Let's assume you already have 'post_centric_graph' and the centrality measures calculated
# Let's first get the top nodes for each centrality measure (Degree, Betweenness, Closeness)

# For Degree Centrality
degree_centrality = nx.degree_centrality(post_centric_graph)
degree_sorted = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)
top_degree_nodes = degree_sorted[:10]  # Top 10 nodes

# For Betweenness Centrality
betweenness_centrality = nx.betweenness_centrality(post_centric_graph)
betweenness_sorted = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)
top_betweenness_nodes = betweenness_sorted[:10]  # Top 10 nodes

# For Closeness Centrality
closeness_centrality = nx.closeness_centrality(post_centric_graph)
closeness_sorted = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)
top_closeness_nodes = closeness_sorted[:10]  # Top 10 nodes

# Combine the top nodes from all measures
top_nodes = set([node for node, _ in top_degree_nodes + top_betweenness_nodes + top_closeness_nodes])

# Inspect the type of each top node (Post or Commenter)
node_types = {}

for node in top_nodes:
    # Check if the node represents a post or a comment (depending on how the data is structured)
    if 'post' in post_centric_graph.nodes[node]:  # You can change this based on your node attributes
        node_types[node] = 'Post'
    else:
        node_types[node] = 'Commenter'

# Create a DataFrame to inspect the top nodes and their types
top_nodes_df = pd.DataFrame(list(node_types.items()), columns=['Node', 'Type'])
print(top_nodes_df)

# Now you can inspect the nodes more closely by retrieving their properties:
# For example, check the post content or the commenter activity:

for node in top_nodes:
    if node_types[node] == 'Post':
        post_data = post_centric_graph.nodes[node]  # Retrieve post data (e.g., content, number of comments)
        print(f"Post Node: {node}, Content: {post_data.get('content', 'No content available')}")
    else:
        commenter_data = post_centric_graph.nodes[node]  # Retrieve commenter data (e.g., number of comments)
        print(f"Commenter Node: {node}, Comments Count: {commenter_data.get('comment_count', 0)}")
        
import matplotlib.pyplot as plt

# Extract degree centrality for posts and commenters
post_degree_centrality = {}
commenter_degree_centrality = {}

for node, centrality in degree_centrality.items():
    if post_centric_graph.nodes[node].get('type') == 'post_author':
        post_degree_centrality[node] = centrality
    elif post_centric_graph.nodes[node].get('type') == 'commenter':
        commenter_degree_centrality[node] = centrality

# Plot degree centrality distributions for posts and commenters
plt.figure(figsize=(12, 6))

# Post degree centrality histogram
plt.subplot(1, 2, 1)
plt.hist(list(post_degree_centrality.values()), bins=30, color='b', alpha=0.7)
plt.title("Degree Centrality Distribution for Posts")
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')

# Commenter degree centrality histogram
plt.subplot(1, 2, 2)
plt.hist(list(commenter_degree_centrality.values()), bins=30, color='g', alpha=0.7)
plt.title("Degree Centrality Distribution for Commenters")
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Extract betweenness centrality for posts and commenters
post_betweenness_centrality = {}
commenter_betweenness_centrality = {}

for node, centrality in betweenness_centrality.items():
    if post_centric_graph.nodes[node].get('type') == 'post_author':
        post_betweenness_centrality[node] = centrality
    elif post_centric_graph.nodes[node].get('type') == 'commenter':
        commenter_betweenness_centrality[node] = centrality

# Plot betweenness centrality distributions for posts and commenters
plt.figure(figsize=(12, 6))

# Post betweenness centrality histogram
plt.subplot(1, 2, 1)
plt.hist(list(post_betweenness_centrality.values()), bins=30, color='b', alpha=0.7)
plt.title("Betweenness Centrality Distribution for Posts")
plt.xlabel('Betweenness Centrality')
plt.ylabel('Frequency')

# Commenter betweenness centrality histogram
plt.subplot(1, 2, 2)
plt.hist(list(commenter_betweenness_centrality.values()), bins=30, color='g', alpha=0.7)
plt.title("Betweenness Centrality Distribution for Commenters")
plt.xlabel('Betweenness Centrality')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Step 1: Initialize the bipartite graph
bipartite_graph = nx.Graph()

# Step 2: Build the bipartite graph
for _, post_data in posts_df.iterrows():
    post_id = post_data['post_id']
    post_author = post_data['author']
    
    # Add post as a node
    bipartite_graph.add_node(post_id, bipartite=0, author=post_author)  # Set the bipartite group for posts
    
    # Get commenters for the current post
    post_comments = comments_df[comments_df['post_id'] == post_id]
    commenters = post_comments['author']
    
    # Add commenters as nodes
    for commenter in commenters:
        if commenter != '[deleted]':  # Skip deleted users
            bipartite_graph.add_node(commenter, bipartite=1)  # Set the bipartite group for commenters
            bipartite_graph.add_edge(post_id, commenter)  # Link post to commenter

# Visualize the graph
import matplotlib.pyplot as plt

# Separate nodes by type
posts = [node for node, data in post_centric_graph.nodes(data=True) if data['type'] == 'post']
commenters = [node for node, data in post_centric_graph.nodes(data=True) if data['type'] == 'commenter']

# Position the nodes (you can experiment with different layouts)
pos = nx.spring_layout(post_centric_graph)

# Draw the graph
plt.figure(figsize=(12, 12))

# Draw posts in one color (e.g., red) and commenters in another color (e.g., blue)
nx.draw_networkx_nodes(post_centric_graph, pos, nodelist=posts, node_color='red', node_size=500)
nx.draw_networkx_nodes(post_centric_graph, pos, nodelist=commenters, node_color='blue', node_size=200)

# Draw edges
nx.draw_networkx_edges(post_centric_graph, pos, alpha=0.5)
plt.show()


# Extract node sets to check if bipartite graph is done correctly
post_nodes = [node for node, data in bipartite_graph.nodes(data=True) if data['bipartite'] == 0]
commenter_nodes = [node for node, data in bipartite_graph.nodes(data=True) if data['bipartite'] == 1]

print(f"Number of post nodes: {len(post_nodes)}")
print(f"Number of commenter nodes: {len(commenter_nodes)}")

#Highlight posts and commenters with distinct colors to verify their placement:
pos = nx.spring_layout(bipartite_graph)
plt.figure(figsize=(10, 10))

nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=post_nodes, node_color='red', node_size=100, label="Posts")
nx.draw_networkx_nodes(bipartite_graph, pos, nodelist=commenter_nodes, node_color='blue', node_size=50, label="Commenters")

nx.draw_networkx_edges(bipartite_graph, pos, alpha=0.5)
plt.legend()
plt.show()
    
# =============================================================================
# #Switch the layout to a bipartite-specific layout like bipartite_layout to explicitly separate the two groups
# pos = nx.bipartite_layout(bipartite_graph, post_nodes)  # Align posts on one side
# plt.figure(figsize=(12, 8))
# 
# nx.draw(bipartite_graph, pos, with_labels=False, node_size=50, alpha=0.7)
# plt.show()
# =============================================================================


filtered_posts = pd.read_csv("../data/onepiece_sentiment_posts_filtered.csv")  # Adjust path if necessary
filtered_comments = pd.read_csv("../data/onepiece_sentiment_comments_filtered.csv")  # Adjust path if necessary

# Step 1: Initialize the bipartite graph
bi_graph_filtered_data = nx.Graph()

# Step 2: Build the bipartite graph
for _, post_data in filtered_posts.iterrows():
    post_id = post_data['post_id']
    post_author = post_data['author_x']
    
    # Add post as a node
    bi_graph_filtered_data.add_node(
        post_id,
        bipartite=0,
        author=post_author,
        type='post'  # Add 'type' attribute for posts
    )
    
    # Get commenters for the current post
    post_comments = filtered_comments[filtered_comments['post_id'] == post_id]
    commenters = post_comments['author']
    
    # Add commenters as nodes
    for commenter in commenters:
        if commenter != '[deleted]':  # Skip deleted users
            bi_graph_filtered_data.add_node(
                commenter,
                bipartite=1,
                type='commenter'  # Add 'type' attribute for commenters
            )
            bi_graph_filtered_data.add_edge(post_id, commenter)  # Link post to commenter

# Visualize the graph
import matplotlib.pyplot as plt

# Separate nodes by type
posts = [node for node, data in bi_graph_filtered_data.nodes(data=True) if data['type'] == 'post']
commenters = [node for node, data in bi_graph_filtered_data.nodes(data=True) if data['type'] == 'commenter']

# Position the nodes (you can experiment with different layouts)
pos = nx.spring_layout(bi_graph_filtered_data)

# Draw the graph
plt.figure(figsize=(12, 12))

# Draw posts in one color (e.g., red) and commenters in another color (e.g., blue)
nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=posts, node_color='red', node_size=500)
nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=commenters, node_color='blue', node_size=200)

# Draw edges
nx.draw_networkx_edges(bi_graph_filtered_data, pos, alpha=0.5)
plt.show()


# Extract node sets to check if bipartite graph is done correctly
post_nodes = [node for node, data in bi_graph_filtered_data.nodes(data=True) if data['bipartite'] == 0]
commenter_nodes = [node for node, data in bi_graph_filtered_data.nodes(data=True) if data['bipartite'] == 1]

print(f"Number of post nodes: {len(post_nodes)}")
print(f"Number of commenter nodes: {len(commenter_nodes)}")

#Highlight posts and commenters with distinct colors to verify their placement:
pos = nx.spring_layout(bi_graph_filtered_data)
plt.figure(figsize=(10, 10))

nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=post_nodes, node_color='red', node_size=100, label="Posts")
nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=commenter_nodes, node_color='blue', node_size=50, label="Commenters")

nx.draw_networkx_edges(bi_graph_filtered_data, pos, alpha=0.5)
plt.legend()
plt.show()




# Use the existing node positions
pos = nx.spring_layout(bi_graph_filtered_data)  # If already computed, reuse this

# Initialize plot
plt.figure(figsize=(15, 15))

# Define colors for edges based on sentiment polarity
edge_colors = []
edge_opacities = []

for u, v in bi_graph_filtered_data.edges():
    # Get sentiment from the filtered_comments DataFrame
    sentiment = filtered_comments.loc[
        (filtered_comments['post_id'] == u) & (filtered_comments['author'] == v),
        'sentiment_body'
    ]
    
    # Skip edges without sentiment data
    if sentiment.empty:
        continue

    sentiment_value = sentiment.values[0]
    
    # Assign color based on sentiment polarity
    if sentiment_value > 0:  # Positive sentiment
        edge_colors.append('green')
    elif sentiment_value < 0:  # Negative sentiment
        edge_colors.append('red')
    else:  # Neutral sentiment
        edge_colors.append('gray')
    
    # Assign opacity based on magnitude of sentiment
    edge_opacities.append(min(1, max(0.1, abs(sentiment_value))))  # Normalize between 0.1 and 1

# Draw nodes
nx.draw_networkx_nodes(
    bi_graph_filtered_data, pos,
    nodelist=posts, node_color='red', node_size=100, label='Posts'
)
nx.draw_networkx_nodes(
    bi_graph_filtered_data, pos,
    nodelist=commenters, node_color='blue', node_size=50, label='Commenters'
)

# Draw edges with colors and opacity
for (u, v), color, opacity in zip(bi_graph_filtered_data.edges(), edge_colors, edge_opacities):
    nx.draw_networkx_edges(
        bi_graph_filtered_data,
        pos,
        edgelist=[(u, v)],
        edge_color=color,
        alpha=opacity,
        width=1
    )

# Add legend
plt.legend()
plt.title("Bipartite Graph with Sentiment Weights (Colored Edges)")
plt.show()


from community import community_louvain

# Apply the Louvain method for community detection
partition = community_louvain.best_partition(bi_graph_filtered_data, weight='weight')

# Analyze the results
print(f"Number of communities detected: {len(set(partition.values()))}")

# Store community information in a dictionary
communities = {}
for node, community in partition.items():
    if community not in communities:
        communities[community] = []
    communities[community].append(node)

# Display some details about the communities
for i, (community, members) in enumerate(communities.items()):
    print(f"Community {i}: {len(members)} members")

# Assign community-based colors
node_colors = [partition[node] for node in bi_graph_filtered_data.nodes()]

# Draw the graph with community coloring
pos = nx.spring_layout(bi_graph_filtered_data)  # Spring layout for better visualization
plt.figure(figsize=(10, 10))
nx.draw_networkx_nodes(
    bi_graph_filtered_data,
    pos,
    node_size=20,
    node_color=node_colors,
    cmap=plt.cm.rainbow  # Use a colormap for better distinction
)
nx.draw_networkx_edges(bi_graph_filtered_data, pos, alpha=0.5)
plt.title("Community Detection with Normalized Weights")
plt.show()



#we try to highlight here the nodes with higher degree centrality
#and as we d expect we can see that they belong to posts


# Compute degree centrality (already done previously)
degree_centrality = nx.degree_centrality(bi_graph_filtered_data)

# Sort nodes by degree centrality in descending order
sorted_degree_centrality = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)

# Extract the top 10 nodes with highest degree centrality
top_10_degree_centrality = [node for node, _ in sorted_degree_centrality[:10]]

# Set node colors: highlight top 10 nodes
node_colors = ['red' if node in top_10_degree_centrality else 'skyblue' for node in bi_graph_filtered_data.nodes()]

# Set node sizes: scale by degree centrality
node_sizes = [5000 * degree_centrality[node] for node in bi_graph_filtered_data.nodes()]

# Plot the graph
plt.figure(figsize=(10, 10))
nx.draw(
    bi_graph_filtered_data,
    pos,
    with_labels=False,  # Remove labels for clarity
    node_size=node_sizes,
    node_color=node_colors,
    alpha=0.8
)
plt.title('Graph Highlighting Top Nodes by Degree Centrality', fontsize=15)
plt.show()


#double check to see if those are actually posts or not

red_nodes = [node for node in top_10_degree_centrality if node in filtered_posts['post_id'].values]
print(f"Red Nodes classified as posts: {red_nodes}")

#so we try to consider only those nodes belonging to 90 percentile

import numpy as np


# Extract degree centrality values
degree_values = list(degree_centrality.values())
# Calculate threshold
threshold = np.percentile(degree_values, 90)  # Adjust percentile as needed
print(f"90th Percentile Threshold: {threshold}")

# Filter nodes based on threshold
central_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]
print(f"Central Nodes: {central_nodes}")

# Create a color map
central_node_colors = ["red" if node in central_nodes else "gray" for node in bi_graph_filtered_data.nodes()]
central_node_sizes = [100 if node in central_nodes else 10 for node in bi_graph_filtered_data.nodes()]

# Plot the graph
plt.figure(figsize=(12, 12))
nx.draw(
    bi_graph_filtered_data,
    pos,
    node_color=central_node_colors,
    node_size=central_node_sizes,
    edge_color=edge_colors,
    alpha=0.7,
    with_labels=False
)
plt.title("Graph Highlighting Central Nodes")
plt.show()



#Plot with central nodes highlighted part2
# Assuming `central_nodes` is a list of nodes above the 90th percentile threshold
# Create a color map for central nodes
central_node_colors = [
    "red" if node in central_nodes else "gray"
    for node in bi_graph_filtered_data.nodes()
]

# Define node sizes with larger scaling for central nodes
central_node_sizes = [
    1000 if node in central_nodes else 10  # Scaling factor for central nodes
    for node in bi_graph_filtered_data.nodes()
]

# Plot the graph
plt.figure(figsize=(15, 15))
nx.draw(
    bi_graph_filtered_data,
    pos,
    node_color=central_node_colors,
    node_size=central_node_sizes,
    edge_color=edge_colors,
    alpha=0.7,
    with_labels=False
)
plt.title("Graph Highlighting Central Nodes (90th Percentile Threshold)")
plt.show()


#plot with central nodes only but this time edges are not coloured
# Plot the graph without edge colors (default edge color)
plt.figure(figsize=(15, 15))
nx.draw(
    bi_graph_filtered_data,
    pos,
    node_color=central_node_colors,
    node_size=central_node_sizes,
    alpha=0.7,
    with_labels=False,
    edge_color='gray'  # Default color for edges (can be changed)
)
plt.title("Graph Highlighting Central Nodes (90th Percentile Threshold)")
plt.show()


#cut the nodes that are not part of the central zone and plot according to community detection

central_subgraph = bi_graph_filtered_data.subgraph(central_nodes)

# Create a new list of node sizes for the subgraph
central_node_sizes_subgraph = [100 if node in central_nodes else 10 for node in central_subgraph.nodes()]

# Extract the community colors for the subgraph (same as before)
subgraph_community_colors = [partition[node] for node in central_subgraph.nodes()]

# Plot the subgraph with community detection applied
plt.figure(figsize=(12, 12))
nx.draw(
    central_subgraph,
    pos,
    node_color=subgraph_community_colors,
    node_size=central_node_sizes_subgraph,  # Using the updated sizes for the subgraph
    alpha=0.7,
    with_labels=False,
    edge_color='gray'
)
plt.title("Zoomed-In View: Central Nodes and Their Communities")
plt.show()


#we try to compute again betweennes and closeness for those nodes belonging to the central area
#in this way we try to compare these distributions to that of degree centrality
#in order to try to better understand if nodes in the central zone are someway important for the network

# Calculate betweenness centrality for central nodes
betweenness_centrality = nx.betweenness_centrality(bi_graph_filtered_data)
# Filter betweenness centrality for the central nodes
central_betweenness = {node: betweenness_centrality[node] for node in central_nodes}

# Calculate closeness centrality for central nodes
closeness_centrality = nx.closeness_centrality(bi_graph_filtered_data)
# Filter closeness centrality for the central nodes
central_closeness = {node: closeness_centrality[node] for node in central_nodes}

# Plot the distributions of betweenness and closeness centrality for central nodes
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


#compute the correlation between degree and other measures in order to check for 
#significance of nodes in the central zone

import numpy as np
from scipy.stats import pearsonr

degree_above_threshold = [node for node, degree in degree_centrality.items() if degree >= threshold]

# Get degree centrality values for nodes above the threshold
central_nodes_degree_values = [degree_centrality[node] for node in degree_above_threshold]
central_nodes_betweenness_values = [betweenness_centrality[node] for node in degree_above_threshold]
central_nodes_closeness_values = [closeness_centrality[node] for node in degree_above_threshold]

# Calculate the Pearson correlation coefficient between degree and betweenness centrality for central zone nodes
degree_betweenness_corr, _ = pearsonr(central_nodes_degree_values, central_nodes_betweenness_values)

# Calculate the Pearson correlation coefficient between degree and closeness centrality for central zone nodes
degree_closeness_corr, _ = pearsonr(central_nodes_degree_values, central_nodes_closeness_values)

# Print the results
print(f"Correlation between degree and betweenness centrality (central zone): {degree_betweenness_corr}")
print(f"Correlation between degree and closeness centrality (central zone): {degree_closeness_corr}")


#at this point we try to cathegorize central nodes to understand their nature

import matplotlib.pyplot as plt

# Placeholder for categorization
node_categories = {
    "post": [],
    "commenter": []
}

# Categorize nodes
for node in central_nodes:
    if node in filtered_posts['post_id'].values:  # Assuming post IDs are in 'post_id'
        node_categories["post"].append(node)
    elif node in filtered_comments['author'].values:  # Assuming commenters are in 'author'
        node_categories["commenter"].append(node)

# Count the occurrences of each category
category_counts = {k: len(v) for k, v in node_categories.items()}

# Print summary
print("Node Categorization Summary:")
for category, count in category_counts.items():
    print(f"{category.capitalize()}: {count}")

# Plot the distribution of node types
plt.bar(category_counts.keys(), category_counts.values(), color=['blue', 'orange'])
plt.xlabel("Node Type")
plt.ylabel("Count")
plt.title("Distribution of High-Centrality Nodes (Post vs Commenter)")
plt.show()

# --- Additional Analysis for Commenters ---
# Calculate the number of comments per high-centrality commenter
high_centrality_commenters = filtered_comments[
    filtered_comments['author'].isin(node_categories["commenter"])
]
commenter_counts = high_centrality_commenters['author'].value_counts()

# Plot the distribution of comments by high-centrality commenters
plt.figure(figsize=(10, 6))
commenter_counts.plot(kind='bar', color='orange')
plt.xlabel("Commenter")
plt.ylabel("Number of Comments")
plt.title("Distribution of Comments by High-Centrality Commenters")
plt.xticks(rotation=45, ha='right', fontsize=1)
plt.tight_layout()
plt.show()


# Re-filter posts based on categorized high-centrality nodes
high_centrality_post_nodes_revised = [
    node for node in central_nodes if node in node_categories["post"]
]

filtered_posts_high_centrality_revised = filtered_posts[
    filtered_posts['post_id'].isin(high_centrality_post_nodes_revised)
]

print("Number of high-centrality posts (revised):", len(filtered_posts_high_centrality_revised))
#Number of high-centrality posts (revised): 18

sorted_posts = filtered_posts_high_centrality_revised.sort_values(
    by='num_comments_x', ascending=False
)

# Extract the post IDs and their corresponding comment counts
post_ids = sorted_posts['post_id']
comment_counts = sorted_posts['num_comments_x']

# Plot the distribution
plt.figure(figsize=(10, 6))
plt.bar(post_ids, comment_counts, color='blue')
plt.xlabel("Post ID")
plt.ylabel("Number of Comments")
plt.title("Distribution of Comments by High-Centrality Posts")
plt.xticks(rotation=45, ha='right', fontsize=10)  # Adjust for readability
plt.tight_layout()
plt.show()

# Sort posts by number of comments in descending order
top_posts = filtered_posts.sort_values(by='num_comments_x', ascending=False)

# Extract relevant features for inspection
top_posts_content = top_posts[['post_id', 'title', 'selftext', 'num_comments_x']]

# Print the top posts
print("Top Posts Content:")
print(top_posts_content)

# Iterate through the top posts and print titles
for index, row in top_posts_content.iterrows():
    print(f"Post ID: {row['post_id']}")
    print(f"Title: {row['title']}")
    print(f"Number of Comments: {row['num_comments_x']}")
    print("=" * 50)  # Separator for better readability


#we try to extract only those commenters which commented more than 10 times as it seems a reasonable number here

# Get the count of comments per unique commenter
comment_count_per_commenter = high_centrality_commenters['author'].value_counts()

# Print the number of comments per commenter
print("Number of comments per commenter:")
print(comment_count_per_commenter.head())  # Show the first few rows for inspection


# Filter top commenters who have made more than 10 comments
top_commenters = comment_count_per_commenter[comment_count_per_commenter > 10]

# Print the top commenters
print(f"Top commenters with more than 10 comments: {len(top_commenters)}")
print(top_commenters.head())  # Show the first few top commenters

# Filter out the rows from high_centrality_commenters that belong to these top commenters
top_commenter_data = high_centrality_commenters[high_centrality_commenters['author'].isin(top_commenters.index)]

# Inspect the filtered data
print(f"Filtered high centrality commenters (with more than 10 comments): {top_commenter_data.shape[0]}")
print(top_commenter_data.head())


# Extract only the usernames
top_commenters_list = top_commenters.index.tolist()

# Ensure they are strings
top_commenters_list = [str(commenter) for commenter in top_commenters_list]

# Normalize `filtered_comments['author']`
filtered_comments['author'] = filtered_comments['author'].str.lower()

# Normalize `top_commenters_list`
top_commenters_list = [commenter.lower().strip() for commenter in top_commenters_list]


top_commenters_data = filtered_comments[
    (filtered_comments['author'].isin(top_commenters_list)) &
    (filtered_comments['post_id'].isin(high_centrality_post_nodes_revised))
]

print("Number of rows in top_commenters_data:", top_commenters_data.shape[0])


commenter_post_counts = top_commenters_data.groupby('author')['post_id'].nunique()
print(commenter_post_counts)

commenter_comment_counts = top_commenters_data['author'].value_counts()
print(commenter_comment_counts)

commenter_activity = pd.DataFrame({
    'unique_posts': commenter_post_counts,
    'total_comments': commenter_comment_counts
})
print(commenter_activity)

import seaborn as sns

# Create a pivot table of comments by post and commenter
pivot_table = top_commenters_data.pivot_table(
    index='author',
    columns='post_id',
    aggfunc='size',
    fill_value=0
)

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='Blues', annot=False)
plt.title("Comments by Top Commenters Across High-Centrality Posts")
plt.ylabel("Top Commenters")
plt.xlabel("Posts")
plt.show()


plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=commenter_activity,
    x='unique_posts',
    y='total_comments',
    hue='total_comments',
    palette='viridis',
    size='total_comments',
    sizes=(20, 200)
)
plt.title("Activity of Top Commenters Across High-Centrality Posts")
plt.xlabel("Unique Posts Engaged")
plt.ylabel("Total Comments")
plt.show()


import pandas as pd

# Create a pivot table from the comments data
heatmap_data = filtered_comments.pivot_table(index='author', columns='post_id', aggfunc='size', fill_value=0)

print("Heatmap data (comments count by author and post):")
print(heatmap_data.head())

# Calculate the sum of comments for each post (i.e., column sums)
post_engagement = heatmap_data.sum(axis=0)

# Define the threshold for high engagement (e.g., top 10% most commented posts)
threshold = post_engagement.quantile(0.9)  # Top 10% threshold

# Select posts with engagement above the threshold
high_engagement_posts = post_engagement[post_engagement > threshold].index

print("High engagement posts (IDs):")
print(high_engagement_posts)

# Get the subset of comments where the post_id is in high engagement posts
high_engagement_comments = filtered_comments[filtered_comments['post_id'].isin(high_engagement_posts)]

# Now, you can analyze the commenters who interacted with those posts
commenter_analysis = high_engagement_comments.groupby('post_id').agg(
    unique_commenters=('author', 'nunique'),
    total_comments=('author', 'count'),
    top_commenter=('author', lambda x: x.value_counts().idxmax()),
    top_commenter_count=('author', lambda x: x.value_counts().max())
).reset_index()

print("Commenter analysis for high engagement posts:")
print(commenter_analysis)



#probably I made a mess modifying the dataframes each time and I need to import again the dataframes needed
filtered_posts = pd.read_csv("../data/onepiece_sentiment_posts_filtered.csv")  # Adjust path if necessary
filtered_comments = pd.read_csv("../data/onepiece_sentiment_comments_filtered.csv")  # Adjust path if necessary

# Check for any discrepancies by comparing top_commenters_list with cleaned version of filtered_comments['author']
filtered_comments['author_clean'] = filtered_comments['author'].str.strip().str.lower()  # clean names

# Check which top commenters are in filtered_comments
top_commenters_clean = [x.lower() for x in top_commenters_list]  # clean top commenters list
missing_commenters = [commenter for commenter in top_commenters_clean if commenter not in filtered_comments['author_clean'].values]

# Display missing commenters
print(f"Missing commenters: {missing_commenters}")

# Check which commenters are present
present_commenters = [commenter for commenter in top_commenters_clean if commenter in filtered_comments['author_clean'].values]
print(f"Present commenters: {present_commenters}")

# Optionally check how many matches we have
print(f"Total matches found: {len(present_commenters)}")


# Initialize an empty graph
G = nx.Graph()

# Add nodes to the graph (top commenters)
G.add_nodes_from(present_commenters)

# Filter comments to include only the top commenters
filtered_top_comments = filtered_comments[filtered_comments['author'].isin(present_commenters)]

# Create edges between commenters who commented on the same post
for post_id in filtered_top_comments['post_id'].unique():
    # Get the commenters for this post
    commenters = filtered_top_comments[filtered_top_comments['post_id'] == post_id]['author'].tolist()
    
    # Add edges between all commenters for this post (if there are at least 2 commenters)
    for i in range(len(commenters)):
        for j in range(i+1, len(commenters)):
            G.add_edge(commenters[i], commenters[j])

# Visualize the network
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, k=0.15, iterations=20)  # Positioning for better visualization
nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')
nx.draw_networkx_edges(G, pos, width=2, alpha=0.5, edge_color='gray')
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
plt.title("Commenter Network - Top Commenters")
plt.show()



# =============================================================================
# # Step 1: Extract the connected components
# connected_components = list(nx.connected_components(G))
# largest_cluster = max(connected_components, key=len)  # Assuming the mini-cluster is the largest connected component
# mini_cluster_graph = G.subgraph(largest_cluster)
# 
# # Step 2: Analyze the cluster's properties
# print(f"Number of nodes in the cluster: {mini_cluster_graph.number_of_nodes()}")
# print(f"Number of edges in the cluster: {mini_cluster_graph.number_of_edges()}")
# print(f"Cluster density: {nx.density(mini_cluster_graph)}")
# 
# # Step 3: Visualize the cluster
# plt.figure(figsize=(8, 8))
# pos = nx.spring_layout(mini_cluster_graph, seed=42)
# nx.draw(
#     mini_cluster_graph, pos,
#     with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=10
# )
# plt.title("Mini Cluster of Commenters")
# plt.show()
# 
# 
# # Calculate centrality measures for the commenters
# centrality = nx.degree_centrality(G)
# centrality_values = {node: centrality[node] for node in G.nodes()}
# 
# # Sort by centrality value to find the most central nodes
# sorted_centrality = sorted(centrality_values.items(), key=lambda item: item[1], reverse=True)
# 
# # Print the top 8 most central commenters
# print("Top 8 most central commenters:", sorted_centrality[:8])
# # =============================================================================
# # Top 8 most central commenters: [('vinsmokewhoswho', 0.3333333333333333), ('kidelaleron', 0.18518518518518517), ('totally_not_a_reply', 0.18518518518518517), ('kerriazes', 0.18518518518518517), ('idkdidkkdkdj', 0.14814814814814814), ('scaptastic', 0.14814814814814814), ('nicentra', 0.14814814814814814), ('hinrik96', 0.14814814814814814)]
# # 
# # =============================================================================
# # Step 4: Identify shared posts that connect commenters in the cluster
# shared_posts = filtered_comments[filtered_comments['author'].isin(largest_cluster)].groupby('post_id')['author'].apply(list)
# shared_posts = shared_posts[shared_posts.apply(len) > 1]  # Keep only posts with multiple commenters
# print("Posts shared by commenters in the mini-cluster:")
# print(shared_posts)
# =============================================================================
# =============================================================================
# Number of nodes in the cluster: 8
# Number of edges in the cluster: 20
# Cluster density: 0.7142857142857143
# 
# =============================================================================


#the following is a unified chunk of code for both the last 2 fragments of code I ran
#its more compact and complete and its outputs different values from the one we just found
#but that is because this time we are only considering nodes belonging to this small network of 8 nodes
#whereas before degree centrality values were computed on the whole network but
#displaying only those 8 nodes of the mini cluster

# Step 1: Extract the largest connected component (mini-cluster)
connected_components = list(nx.connected_components(G))
largest_cluster = max(connected_components, key=len)
mini_cluster_graph = G.subgraph(largest_cluster)

# Step 2: Analyze the mini-cluster properties
print(f"Number of nodes in the cluster: {mini_cluster_graph.number_of_nodes()}")
print(f"Number of edges in the cluster: {mini_cluster_graph.number_of_edges()}")
print(f"Cluster density: {nx.density(mini_cluster_graph)}")

# Step 3: Visualize the mini-cluster
plt.figure(figsize=(8, 8))
pos = nx.spring_layout(mini_cluster_graph, seed=42)
nx.draw(
    mini_cluster_graph, pos,
    with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=10
)
plt.title("Mini Cluster of Commenters")
plt.show()

# Step 4: Centrality analysis of the cluster
degree_centrality = nx.degree_centrality(mini_cluster_graph)
betweenness_centrality = nx.betweenness_centrality(mini_cluster_graph)
closeness_centrality = nx.closeness_centrality(mini_cluster_graph)

# Combine centralities into a dataframe for better analysis
import pandas as pd
centrality_df = pd.DataFrame({
    'degree': degree_centrality,
    'betweenness': betweenness_centrality,
    'closeness': closeness_centrality
}).sort_values(by='degree', ascending=False)

print("Centrality measures for the mini-cluster:")
print(centrality_df)

# Step 5: Identify shared posts
shared_posts = filtered_comments[filtered_comments['author'].isin(largest_cluster)].groupby('post_id')['author'].apply(list)
shared_posts = shared_posts[shared_posts.apply(len) > 1]  # Only posts with multiple commenters
print("Posts shared by commenters in the mini-cluster:")
print(shared_posts)

# Optional: Fetch post titles for shared posts (if available in filtered_posts)
shared_post_ids = shared_posts.index
shared_post_titles = filtered_posts[filtered_posts['post_id'].isin(shared_post_ids)][['post_id', 'title']]
print("Titles of shared posts:")
print(shared_post_titles)



# List of mini-cluster commenters (update if needed)
mini_cluster_commenters = ['vinsmokewhoswho', 'kidelaleron', 'totally_not_a_reply', 'kerriazes',
                           'idkdidkkdkdj', 'scaptastic', 'nicentra', 'hinrik96']

# Filter comments for the mini-cluster
mini_cluster_comments = filtered_comments[filtered_comments['author'].isin(mini_cluster_commenters)]

# Count sentiment categories for the mini-cluster
sentiment_counts = mini_cluster_comments['sentiment_category'].value_counts()

# Create a pie chart
plt.figure(figsize=(8, 8))
colors = ['#ff9999', '#66b3ff', '#99ff99']  # Colors for Positive, Neutral, Negative
sentiment_counts.plot.pie(
    autopct='%1.1f%%', 
    startangle=90, 
    colors=colors, 
    labels=sentiment_counts.index
)
plt.title("Sentiment Distribution for Mini-Cluster Commenters")
plt.ylabel('')  # Remove y-axis label for aesthetics
plt.show()


# Assuming filtered_comments contains 'author' and 'created_utc' columns
# Also assuming you have a list 'mini_cluster_commenters' with the 8 commenters

# Convert 'created_utc' to datetime (this assumes 'created_utc' is in UNIX timestamp format)
filtered_comments['created_utc'] = pd.to_datetime(filtered_comments['created_utc'], unit='s')

# Filter only the mini-cluster commenters
mini_cluster_comments = filtered_comments[filtered_comments['author'].isin(mini_cluster_commenters)]

# Extract date from 'created_utc'
mini_cluster_comments['date'] = mini_cluster_comments['created_utc'].dt.date

# Group by date and count comments
daily_activity = mini_cluster_comments.groupby('date').size()

# Plotting
plt.figure(figsize=(10, 6))
daily_activity.plot(kind='line', marker='o', color='b')
plt.title('Daily Comment Activity of Top 8 Commenters')
plt.xlabel('Date')
plt.ylabel('Number of Comments')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Assuming filtered_comments already has 'created_utc' and 'sentiment_category'

# Convert 'created_utc' to datetime if it's not already in datetime format
filtered_comments['created_utc'] = pd.to_datetime(filtered_comments['created_utc'], unit='s')

# Extract a period of time (e.g., day, week, month) for grouping
# Here we'll group by day for a clearer trend visualization, but you can adjust this
filtered_comments['date'] = filtered_comments['created_utc'].dt.date

# Now group by 'date' and 'sentiment_category' to count occurrences
sentiment_by_date = filtered_comments.groupby(['date', 'sentiment_category']).size().unstack(fill_value=0)

# Visualize the trends over time for each sentiment category
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
sentiment_by_date.plot(kind='line', marker='o', color=['#66c2a5', '#fc8d62', '#8da0cb'])
plt.title('Sentiment Trends Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Comments')
plt.legend(title='Sentiment Category', labels=['Positive', 'Neutral', 'Negative'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Step 1: Extract sentiment proportions for mini-cluster commenters
mini_cluster_comments_data = filtered_comments[filtered_comments['author_clean'].isin(mini_cluster_commenters)]

# Calculate sentiment proportions
sentiment_proportions = (
    mini_cluster_comments_data
    .groupby(['author_clean', 'sentiment_category'])
    .size()
    .unstack(fill_value=0)
)

# Normalize proportions by user
sentiment_proportions = sentiment_proportions.div(
    sentiment_proportions.sum(axis=1), axis=0
)

# Ensure we're dealing with the correct common authors
common_authors = sentiment_proportions.index

# Aggregate sentiment data by author (if it's not already aggregated)
# Let's focus on the 'Positive' sentiment category for now
sentiment_data = sentiment_proportions.loc[common_authors, 'Positive']

# Get centrality values for each author (degree, betweenness, closeness)
degree_data = pd.Series(degree_centrality).loc[common_authors]
betweenness_data = pd.Series(betweenness_centrality).loc[common_authors]
closeness_data = pd.Series(closeness_centrality).loc[common_authors]

# Now let's create the plots for each centrality measure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Loop through the centrality measures
for i, (centrality_data, centrality_name) in enumerate(zip(
    [degree_data, betweenness_data, closeness_data], 
    ['Degree', 'Betweenness', 'Closeness']
)):
    ax = axes[i]
    
    # Scatter plot for the 'Positive' sentiment category
    ax.scatter(centrality_data, sentiment_data, color='green', alpha=0.7, label='Positive')
    
    # Add labels and title
    ax.set_xlabel(f'{centrality_name} Centrality')
    ax.set_ylabel('Positive Sentiment Proportion')
    ax.set_title(f'{centrality_name} Centrality vs Positive Sentiment')
    ax.legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


from collections import Counter

# Assuming bi_graph_filtered_data is the graph and filtered_comments is the dataframe
# containing 'author_clean' and 'sentiment_category' columns

# Step 1: Calculate the dominant sentiment for each commenter
# Group by 'author_clean' and calculate the most common sentiment for each author
sentiment_counts = filtered_comments.groupby('author_clean')['sentiment_category'].apply(lambda x: Counter(x).most_common(1))

# Convert sentiment_counts into a dictionary where keys are author names and values are the dominant sentiment
dominant_sentiments = {author: sentiment[0][0] for author, sentiment in sentiment_counts.items()}

# Handling ties by assigning "Neutral" sentiment for authors with a tie between sentiments
for author, sentiment in sentiment_counts.items():
    if len(sentiment) > 1:
        dominant_sentiments[author] = 'Neutral'  # Resolving ties

# Step 2: Compute sentiment assortativity
# Extract the sentiment of each node (commenter)
sentiments_of_nodes = []
for node in bi_graph_filtered_data.nodes:
    if node in dominant_sentiments:
        sentiments_of_nodes.append(dominant_sentiments[node])

# Remove nodes with no sentiment assigned (None values)
sentiments_of_nodes = [sentiment for sentiment in sentiments_of_nodes if sentiment is not None]

# Step 3: Compute sentiment assortativity coefficient using NetworkX
# Create a mapping from author names to sentiment for each node in the graph
sentiment_dict = {node: dominant_sentiments.get(node, 'Neutral') for node in bi_graph_filtered_data.nodes}

# Use NetworkX to calculate the assortativity coefficient for sentiment
sentiment_assortativity = nx.assortativity.attribute_assortativity_coefficient(bi_graph_filtered_data, 'sentiment_category')

# Step 4: Print the results
print("Sentiment Assortativity Coefficient:", sentiment_assortativity)

# Optional: Visualize sentiment distribution
sentiment_distribution = Counter(sentiments_of_nodes)
print("Sentiment Distribution:", sentiment_distribution)

# Step 5: Print the dominant sentiment for the first 10 commenters (for inspection)
print("Dominant Sentiments for the first 10 commenters:")
for i, (author, sentiment) in enumerate(dominant_sentiments.items()):
    if i == 10: break
    print(f"{author}: {sentiment}")


from collections import Counter, defaultdict

# Step 1: Initialize the network with sentiments
# Assign initial sentiments to each node
nx.set_node_attributes(bi_graph_filtered_data, dominant_sentiments, "sentiment")

# Step 2: Define sentiment propagation function
def propagate_sentiments(graph, max_steps=10):
    """
    Simulates sentiment propagation in the network.

    Args:
        graph: A NetworkX graph with 'sentiment' as a node attribute.
        max_steps: Maximum number of propagation steps.

    Returns:
        sentiment_over_time: A list of sentiment distributions at each step.
    """
    sentiment_over_time = []

    # Get the initial sentiment distribution
    initial_sentiments = nx.get_node_attributes(graph, "sentiment")
    sentiment_over_time.append(Counter(initial_sentiments.values()))

    for step in range(max_steps):
        new_sentiments = {}
        # Update sentiments for each node based on neighbors
        for node in graph.nodes:
            neighbors = list(graph.neighbors(node))
            if not neighbors:  # Skip isolated nodes
                new_sentiments[node] = graph.nodes[node]['sentiment']
                continue

            # Get the sentiments of neighbors
            neighbor_sentiments = [
                graph.nodes[neighbor].get("sentiment", "Neutral")
                for neighbor in neighbors
            ]
            # Determine the new sentiment as the majority sentiment
            new_sentiment = Counter(neighbor_sentiments).most_common(1)[0][0]
            new_sentiments[node] = new_sentiment

        # Apply the new sentiments to the graph
        nx.set_node_attributes(graph, new_sentiments, "sentiment")

        # Calculate the new sentiment distribution
        current_sentiments = nx.get_node_attributes(graph, "sentiment")
        sentiment_over_time.append(Counter(current_sentiments.values()))

        # Check for convergence (no changes in sentiments)
        if sentiment_over_time[-1] == sentiment_over_time[-2]:
            print(f"Converged at step {step + 1}")
            break

    return sentiment_over_time

# Step 3: Run the sentiment propagation model
sentiment_evolution = propagate_sentiments(bi_graph_filtered_data)

# Step 4: Visualize the results
# Extract proportions for each sentiment at each time step
sentiment_labels = ["Positive", "Negative", "Neutral"]
proportions = {label: [] for label in sentiment_labels}

for distribution in sentiment_evolution:
    total = sum(distribution.values())
    for label in sentiment_labels:
        proportions[label].append(distribution.get(label, 0) / total)

# Plot the evolution of sentiments
plt.figure(figsize=(10, 6))
for label, values in proportions.items():
    plt.plot(values, label=label)
plt.xlabel("Time Step")
plt.ylabel("Proportion of Sentiments")
plt.title("Sentiment Propagation Over Time")
plt.legend()
plt.grid(True)
plt.show()
#Converged at step 2
# =============================================================================
# The fact that the process terminates in just 2 iterations could mean:
# 
# Fast Convergence Due to Network Topology:
# 
# If high-degree nodes dominate the network and most of their neighbors quickly adopt their sentiment, this rapid convergence might be expected.
# However, this could indicate a lack of sufficient complexity or diversity in the propagation mechanism.
# If all nodes rapidly adopt the same sentiment, it suggests the propagation model might not account for dissenting influences or randomness.
# Rigidity in Sentiment Influence:
# 
# Without incorporating weights, randomness, or external influences, the propagation model might overly simplify the dynamics.
# For instance, nodes might immediately "agree" with the majority sentiment in their neighborhood, creating a rapid consensus.
# 
# =============================================================================


#check the average degree per sentiment to justify the propagation of positive sentiments just found
# =============================================================================
# # Get dominant sentiment for each commenter
# node_sentiments = {node: dominant_sentiments.get(node, "None") for node in bi_graph_filtered_data.nodes()}
# 
# # Compute degree for each node
# node_degrees = dict(bi_graph_filtered_data.degree())
# 
# # Group nodes by sentiment
# sentiment_groups = {"Positive": [], "Neutral": [], "Negative": []}
# for node, sentiment in node_sentiments.items():
#     if sentiment in sentiment_groups:
#         sentiment_groups[sentiment].append(node_degrees[node])
# 
# # Calculate average degree for each sentiment
# average_degrees = {
#     sentiment: sum(degrees) / len(degrees) if degrees else 0
#     for sentiment, degrees in sentiment_groups.items()
# }
# 
# print("Average Degree by Sentiment:")
# for sentiment, avg_degree in average_degrees.items():
#     print(f"{sentiment}: {avg_degree:.2f}")
#     
# Average Degree by Sentiment:
# Positive: 1.11
# Neutral: 1.17
# Negative: 1.09
# =============================================================================


import networkx as nx
import pandas as pd
from collections import Counter
import random
import matplotlib.pyplot as plt

# ===== Step 1: Build the Commenter-Commenter Network =====
# Initialize the commenter-commenter graph
commenter_graph = nx.Graph()

# Group comments by post and create edges between commenters who commented on the same post
for post_id, post_comments in filtered_comments.groupby('post_id'):
    commenters = post_comments['author'].unique()
    
    # Create edges between all pairs of commenters
    for i in range(len(commenters)):
        for j in range(i + 1, len(commenters)):
            commenter1, commenter2 = commenters[i], commenters[j]
            
            # Initialize or update edge weight based on shared posts
            if not commenter_graph.has_edge(commenter1, commenter2):
                commenter_graph.add_edge(commenter1, commenter2, weight=0)
            commenter_graph[commenter1][commenter2]['weight'] += 1

# ===== Step 2: Assign Initial Sentiments =====
# Re-assign initial sentiments based on the average sentiment_body of each commenter
def assign_initial_sentiment(row):
    if row > 0:
        return 1  # Positive
    elif row < 0:
        return -1  # Negative
    else:
        return 0  # Neutral

# Aggregate sentiment_body by author and apply the sentiment assignment function
initial_sentiments = filtered_comments.groupby('author')['sentiment_body'].mean()
initial_sentiments = initial_sentiments.apply(assign_initial_sentiment)

# Assign the sentiments to the graph nodes
nx.set_node_attributes(commenter_graph, initial_sentiments.to_dict(), name='sentiment')

# Print the initial sentiment distribution
initial_distribution = Counter(nx.get_node_attributes(commenter_graph, 'sentiment').values())
print("Initial Sentiment Distribution:")
print(f"Positive: {initial_distribution[1]}")
print(f"Neutral: {initial_distribution[0]}")
print(f"Negative: {initial_distribution[-1]}")

import random
import numpy as np

random.seed(42)
np.random.seed(42)


# ===== Step 3: Simulate Sentiment Propagation =====
#change resistance prob and flip prob to change dynamics of the evolution
def propagate_sentiments(graph, max_steps=10, resistance_prob=0.05, flip_prob=0.1):
    """
    Simulate sentiment propagation on the commenter graph.
    """
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
            if weighted_sum > 0:
                new_sentiments[node] = 1  # Positive
            elif weighted_sum < 0:
                new_sentiments[node] = -1  # Negative
            else:
                new_sentiments[node] = 0  # Neutral

        # Update the graph with new sentiments
        nx.set_node_attributes(graph, new_sentiments, 'sentiment')
        sentiment_distribution = Counter(new_sentiments.values())
        sentiment_over_time.append(sentiment_distribution)

        # Check for convergence
        if sentiment_over_time[-1] == sentiment_over_time[-2]:
            print(f"Converged at step {step + 1}")
            break

    return sentiment_over_time


# Backup original sentiments before running the propagation
original_sentiments = nx.get_node_attributes(commenter_graph, 'sentiment')

# Restore the original sentiments
nx.set_node_attributes(commenter_graph, original_sentiments, 'sentiment')

# Run sentiment propagation
#change number of steps 
sentiment_evolution = propagate_sentiments(commenter_graph, max_steps=50)

# ===== Step 4: Visualize Sentiment Propagation =====
# Extract sentiment proportions over time
sentiment_labels = [-1, 0, 1]  # Negative, Neutral, Positive
proportions = {label: [] for label in sentiment_labels}

for distribution in sentiment_evolution:
    total = sum(distribution.values())
    for label in sentiment_labels:
        proportions[label].append(distribution.get(label, 0) / total)

# Plot sentiment evolution over time
plt.figure(figsize=(10, 6))
for label, values in proportions.items():
    plt.plot(values, label={-1: "Negative", 0: "Neutral", 1: "Positive"}[label])
plt.xlabel("Time Step")
plt.ylabel("Proportion of Sentiments")
plt.title("Sentiment Propagation Over Time")
plt.legend()
plt.grid(True)
plt.show()


#numerical check to understand if the system has converged in someway after the iterations
# Calculate absolute differences between consecutive steps
convergence_threshold = 0.001  # Define a small threshold for convergence
converged = True

for t in range(1, len(sentiment_evolution)):
    prev_dist = sentiment_evolution[t - 1]
    curr_dist = sentiment_evolution[t]

    total_nodes = sum(prev_dist.values())  # Normalize by total number of nodes
    diffs = [
        abs(curr_dist.get(label, 0) / total_nodes - prev_dist.get(label, 0) / total_nodes)
        for label in [-1, 0, 1]  # Sentiment labels
    ]

    # Check if all differences are below the threshold
    if any(diff > convergence_threshold for diff in diffs):
        converged = False
        break

# Print the result
if converged:
    print("The system has reached convergence.")
else:
    print("The system has NOT converged yet.")
    

#the following code is made to understand if there are communities in the graph and understand
#why we dont really see a converging pattern in the sentiment propagation

import copy
from community import community_louvain
from collections import Counter

# Create a temporary copy of the graph for weight shifting
shifted_graph = copy.deepcopy(commenter_graph)

# Shift edge weights to make them non-negative
min_weight = min(d['weight'] for _, _, d in shifted_graph.edges(data=True))
for u, v, d in shifted_graph.edges(data=True):
    d['weight'] += abs(min_weight)

# Perform community detection on the shifted graph
partition = community_louvain.best_partition(shifted_graph)

# Analyze and print community sizes
community_sizes = Counter(partition.values())
print("Community Sizes (number of nodes per community):")
for community, size in community_sizes.items():
    print(f"Community {community}: {size} nodes")

# Optional: Assign communities as node attributes for visualization or further analysis
nx.set_node_attributes(commenter_graph, partition, name='community')

#now we inspect each community to understand how they are made
# Aggregate sentiment distribution for each community
community_sentiments = {c: Counter() for c in community_sizes.keys()}

# Populate sentiment counts
for node, community in partition.items():
    sentiment = commenter_graph.nodes[node]['sentiment']
    community_sentiments[community][sentiment] += 1

# Print sentiment distribution per community
print("Sentiment Distribution per Community:")
for community, sentiments in community_sentiments.items():
    total = sum(sentiments.values())
    print(f"Community {community}:")
    for sentiment, count in sentiments.items():
        proportion = count / total
        label = {1: "Positive", 0: "Neutral", -1: "Negative"}[sentiment]
        print(f"  {label}: {count} ({proportion:.2%})")


#community detection in the commenter network to understand why we didnt reach proper convergence in
#sentiment propagtion

# Generate a spring layout for better separation of communities
pos = nx.spring_layout(commenter_graph, seed=42)

# Assign colors to each community
num_communities = len(set(partition.values()))
color_map = plt.cm.rainbow  # Choose a colormap
colors = [color_map(i / num_communities) for i in range(num_communities)]

# Create a dictionary mapping communities to colors
community_colors = {community: colors[i] for i, community in enumerate(set(partition.values()))}

# Assign colors to nodes based on their community
node_colors = [community_colors[partition[node]] for node in commenter_graph.nodes]

# Create the figure and axes
fig, ax = plt.subplots(figsize=(12, 12))

# Visualize the graph
nx.draw_networkx(
    commenter_graph,
    pos=pos,
    with_labels=False,
    node_size=30,
    node_color=node_colors,
    edge_color='gray',
    alpha=0.7,
    ax=ax  # Link the drawing to the axes
)

# Add a colorbar to the plot
sm = plt.cm.ScalarMappable(cmap=color_map, norm=plt.Normalize(vmin=0, vmax=num_communities - 1))
sm.set_array([])  # Empty array needed for the colorbar to work
fig.colorbar(sm, ax=ax, label="Community")  # Add the colorbar to the specific axes

plt.title("Commenter Network Colored by Communities", fontsize=16)
plt.show()


# Compute degree centrality to rank nodes by connectivity
degree_centrality = nx.degree_centrality(commenter_graph)

# Select top 500 commenters by degree centrality
top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:500]
top_node_ids = [node for node, _ in top_nodes]

# Create subgraph with only top nodes
subgraph = commenter_graph.subgraph(top_node_ids)

print(f"Top Commenters Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
#Top Commenters Subgraph: 500 nodes, 47038 edges

# Compute degree centrality
degree_centrality = nx.degree_centrality(subgraph)

# Get top 5 nodes by degree centrality
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top Influencers by Degree Centrality:", top_degree)
#Top Influencers by Degree Centrality: [('Dillo64', 0.8196392785571142), ('vinsmokewhoswho', 0.8096192384769538), ('Sawgon', 0.7935871743486973), ('Kirosh2', 0.7434869739478958), ('RemindMeBot', 0.7294589178356713)]


# Compute betweenness centrality
betweenness_centrality = nx.betweenness_centrality(subgraph, normalized=True)
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top Influencers by Betweenness Centrality:", top_betweenness)
#Top Influencers by Betweenness Centrality: [('Dillo64', 0.010923494713298823), ('Sawgon', 0.009240525296300785), ('Kirosh2', 0.008463990466315971), ('vinsmokewhoswho', 0.00830125919515779), ('RemindMeBot', 0.007705813073736331)]



# Compute eigenvector centrality
eigenvector_centrality = nx.eigenvector_centrality(subgraph)
top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top Influencers by Eigenvector Centrality:", top_eigenvector)
#Top Influencers by Eigenvector Centrality: [('vinsmokewhoswho', 0.08936522612553033), ('Dillo64', 0.08572693476515704), ('Sawgon', 0.08442841949930126), ('Bohzee', 0.08102860673868853), ('Popopirat66', 0.08000251999538853)]


# Calculate sentiment influence scores
influence_scores = {node: 0 for node in subgraph.nodes}

for node in subgraph.nodes:
    for neighbor in subgraph.neighbors(node):
        if subgraph.nodes[node]['sentiment'] == subgraph.nodes[neighbor]['sentiment']:
            influence_scores[node] += 1

# Get top 5 nodes by sentiment influence
top_sentiment_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top Influencers by Sentiment Influence:", top_sentiment_influencers)
#Top Influencers by Sentiment Influence: [('vinsmokewhoswho', 258), ('Dillo64', 247), ('Popopirat66', 229), ('Asian_Persuasion_1', 229), ('Kirosh2', 222)]


# Get community assignments for top nodes
top_nodes = {node for node, _ in (top_degree + top_betweenness + top_eigenvector + top_sentiment_influencers)}

print("Community Assignments of Top Nodes:")
for node in top_nodes:
    print(f"Node {node} - Community: {partition[node]}")
    
# =============================================================================
# Community Assignments of Top Nodes:
# Node Asian_Persuasion_1 - Community: 8
# Node Bohzee - Community: 1
# Node Kirosh2 - Community: 5
# Node RemindMeBot - Community: 11
# Node Popopirat66 - Community: 6
# Node Sawgon - Community: 5
# Node Dillo64 - Community: 1
# Node vinsmokewhoswho - Community: 6
# =============================================================================


# Create subgraph of top nodes
top_node_subgraph = subgraph.subgraph(top_nodes)

# Print connectivity stats
print(f"Top Nodes Subgraph: {top_node_subgraph.number_of_nodes()} nodes, {top_node_subgraph.number_of_edges()} edges")
print("Is the subgraph fully connected?", nx.is_connected(top_node_subgraph))
# =============================================================================
# Top Nodes Subgraph: 8 nodes, 27 edges
# Is the subgraph fully connected? True
# =============================================================================

#Top nodes highlighted by community
node_sizes = [500 if node in top_nodes else 30 for node in subgraph.nodes]
node_colors = [community_colors[partition[node]] for node in subgraph.nodes]

plt.figure(figsize=(12, 12))
nx.draw_networkx(
    subgraph,
    pos=nx.spring_layout(subgraph, seed=42),
    with_labels=False,
    node_size=node_sizes,
    node_color=node_colors,
    edge_color='gray',
    alpha=0.7
)
plt.title("Top Nodes Highlighted by Community", fontsize=16)
plt.show()

# =============================================================================
# # Analyze each community
# for community in set(partition.values()):
#     # Subset graph for the community
#     community_nodes = [node for node in subgraph.nodes if partition[node] == community]
#     community_subgraph = subgraph.subgraph(community_nodes)
# 
#     # Skip empty subgraphs
#     if community_subgraph.number_of_nodes() == 0:
#         print(f"Skipping empty community {community}")
#         continue
# =============================================================================

# Analyze each community
for community in set(partition.values()):
    # Subset graph for the community
    community_nodes = [node for node in subgraph.nodes if partition[node] == community]
    community_subgraph = subgraph.subgraph(community_nodes)

    # Skip invalid subgraphs (empty or no edges)
    if community_subgraph.number_of_nodes() == 0 or community_subgraph.number_of_edges() == 0:
        print(f"Skipping community {community} due to insufficient structure.")
        continue

    # Store top influencers per community
    community_influencers = defaultdict(dict)

    # Compute centrality measures for the community
    degree_centrality = nx.degree_centrality(community_subgraph)
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    community_influencers[community]['degree'] = top_degree

    betweenness_centrality = nx.betweenness_centrality(community_subgraph)
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
    community_influencers[community]['betweenness'] = top_betweenness

    try:
        # Compute eigenvector centrality if the graph is valid
        eigenvector_centrality = nx.eigenvector_centrality(community_subgraph)
        top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        community_influencers[community]['eigenvector'] = top_eigenvector
    except nx.NetworkXError as e:
        print(f"Skipping community {community} due to centrality computation error: {e}")


# Store top influencers per community
community_influencers = defaultdict(dict)

# Compute centrality measures for the community
degree_centrality = nx.degree_centrality(community_subgraph)
top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
community_influencers[community]['degree'] = top_degree

betweenness_centrality = nx.betweenness_centrality(community_subgraph)
top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
community_influencers[community]['betweenness'] = top_betweenness
# =============================================================================
# # Check if the subgraph is valid
# if community_subgraph.number_of_nodes() == 0 or community_subgraph.number_of_edges() == 0:
#     print(f"Skipping community {community} due to insufficient structure.")
#     continue
# 
# try:
#     # Compute eigenvector centrality if the graph is valid
#     eigenvector_centrality = nx.eigenvector_centrality(community_subgraph)
#     top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
#     community_influencers[community]['eigenvector'] = top_eigenvector
# except nx.NetworkXError as e:
#     print(f"Skipping community {community} due to centrality computation error: {e}")
# =============================================================================

 
eigenvector_centrality = nx.eigenvector_centrality(community_subgraph)
top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
community_influencers[community]['eigenvector'] = top_eigenvector
    
# =============================================================================
# Skipping empty community 9
# Skipping empty community 12
# Skipping empty community 13
# Skipping empty community 14
# Skipping empty community 15
# Skipping empty community 16
# 
# =============================================================================


# Check the number of nodes in each community in the subgraph
filtered_community_sizes = Counter(partition[node] for node in subgraph.nodes)
print("Filtered Community Sizes:", filtered_community_sizes)
#Filtered Community Sizes: Counter({8: 108, 1: 103, 6: 78, 3: 75, 2: 63, 5: 42, 11: 16, 4: 6, 10: 4, 7: 3, 0: 2})


# Calculate sentiment distribution for the largest communities
for community in [8, 1, 6, 3]:
    community_nodes = [node for node in subgraph.nodes if partition[node] == community]
    sentiments = [subgraph.nodes[node]['sentiment'] for node in community_nodes]
    sentiment_counts = Counter(sentiments)
    total = sum(sentiment_counts.values())
    print(f"Community {community}:")
    for sentiment, count in sentiment_counts.items():
        label = {1: "Positive", 0: "Neutral", -1: "Negative"}[sentiment]
        print(f"  {label}: {count} ({count / total:.2%})")
        
# =============================================================================
# Community 8:
#   Positive: 51 (47.22%)
#   Negative: 45 (41.67%)
#   Neutral: 12 (11.11%)
# Community 1:
#   Negative: 26 (25.24%)
#   Positive: 70 (67.96%)
#   Neutral: 7 (6.80%)
# Community 6:
#   Positive: 63 (80.77%)
#   Negative: 13 (16.67%)
#   Neutral: 2 (2.56%)
# Community 3:
#   Negative: 18 (24.00%)
#   Positive: 45 (60.00%)
#   Neutral: 12 (16.00%)
# 
# =============================================================================






import community as community_louvain
modularity = community_louvain.modularity(partition, commenter_graph)
print(f"Modularity Score: {modularity}")
#Modularity Score: 0.7271046523083083


# Initialize a matrix for sentiment flow between communities in the full graph
num_communities_full = len(set(partition.values()))  # Total communities from Louvain
full_sentiment_flow_matrix = np.zeros((num_communities_full, num_communities_full))

# Iterate over edges in the full graph
for u, v, d in commenter_graph.edges(data=True):  # Use full graph
    community_u = partition[u]
    community_v = partition[v]
    weight = d.get('weight', 1)  # Default weight to 1 if not set
    sentiment_u = commenter_graph.nodes[u]['sentiment']
    sentiment_v = commenter_graph.nodes[v]['sentiment']

    # Increment the flow based on sentiment alignment
    if sentiment_u == sentiment_v:
        full_sentiment_flow_matrix[community_u, community_v] += weight

# Print the recomputed matrix
print("Full Sentiment Flow Matrix:")
print(full_sentiment_flow_matrix)


# Heatmap visualization of full sentiment flow matrix
import seaborn as sns
plt.figure(figsize=(12, 10))
sns.heatmap(full_sentiment_flow_matrix, annot=True, fmt=".1f", cmap="coolwarm")
plt.title("Full Sentiment Flow Matrix Heatmap")
plt.xlabel("Community")
plt.ylabel("Community")
plt.show()


# Extract off-diagonal elements (inter-community flows)
num_communities = full_sentiment_flow_matrix.shape[0]
inter_community_flows = [
    (i, j, full_sentiment_flow_matrix[i, j])
    for i in range(num_communities)
    for j in range(num_communities)
    if i != j and full_sentiment_flow_matrix[i, j] > 0  # Exclude diagonal and negative values
]

# Sort flows by value in descending order
inter_community_flows = sorted(inter_community_flows, key=lambda x: x[2], reverse=True)

# Print the top inter-community flows
print("Top Inter-Community Flows:")
for i, j, flow in inter_community_flows[:10]:  # Top 10 flows
    print(f"Community {i} ↔ Community {j}: {flow:.2f}")

# =============================================================================
# Top Inter-Community Flows:
# Community 3 ↔ Community 2: 886.09
# Community 1 ↔ Community 2: 696.68
# Community 8 ↔ Community 2: 545.33
# Community 2 ↔ Community 1: 466.32
# Community 3 ↔ Community 1: 382.81
# Community 3 ↔ Community 6: 361.55
# Community 1 ↔ Community 10: 304.04
# Community 1 ↔ Community 6: 275.04
# Community 2 ↔ Community 7: 272.48
# Community 11 ↔ Community 8: 221.03
# =============================================================================

# Collect nodes involved in top inter-community flows
bridge_nodes = set()
for i, j, _ in inter_community_flows[:10]:  # Top 10 inter-community flows
    community_i_nodes = [node for node in commenter_graph.nodes if partition[node] == i]
    community_j_nodes = [node for node in commenter_graph.nodes if partition[node] == j]
    for node in community_i_nodes:
        for neighbor in commenter_graph.neighbors(node):
            if neighbor in community_j_nodes:
                bridge_nodes.add(node)
                bridge_nodes.add(neighbor)

# Compute degree centrality for these nodes
degree_centrality = nx.degree_centrality(commenter_graph)
bridge_node_scores = [(node, degree_centrality[node]) for node in bridge_nodes]

# Get top 10 bridge nodes by degree centrality
top_bridge_nodes = sorted(bridge_node_scores, key=lambda x: x[1], reverse=True)[:10]
print("Top Bridge Nodes by Degree Centrality:")
for node, score in top_bridge_nodes:
    print(f"Node {node}: {score:.4f}")
# =============================================================================
#     
# Top Bridge Nodes by Degree Centrality:
# Node Dillo64: 0.5015
# Node MariJoyBoy: 0.4448
# Node RemindMeBot: 0.4381
# Node vinsmokewhoswho: 0.4357
# Node Kuro013: 0.3902
# Node Asian_Persuasion_1: 0.3893
# Node Kiosade: 0.3794
# Node ammarbadhrul: 0.3611
# Node Dj0sh: 0.3329
# Node MaddestChadLad: 0.3140
#

# Analyze sentiment distribution for each community
community_sentiments = {}
for community in set(partition.values()):
    community_nodes = [node for node in commenter_graph.nodes if partition[node] == community]
    sentiments = [commenter_graph.nodes[node]['sentiment'] for node in community_nodes]
    sentiment_counts = Counter(sentiments)
    total = sum(sentiment_counts.values())
    community_sentiments[community] = {
        sentiment: count / total for sentiment, count in sentiment_counts.items()
    }

# Print sentiment distributions per community
for community, distribution in community_sentiments.items():
    print(f"Community {community}:")
    for sentiment, proportion in distribution.items():
        label = {1: "Positive", 0: "Neutral", -1: "Negative"}[sentiment]
        print(f"  {label}: {proportion:.2%}")
        
# =============================================================================
# Community 0:
#   Negative: 23.97%
#   Neutral: 31.16%
#   Positive: 44.86%
# Community 1:
#   Positive: 65.43%
#   Negative: 21.59%
#   Neutral: 12.98%
# Community 2:
#   Negative: 22.17%
#   Positive: 62.75%
#   Neutral: 15.08%
# Community 3:
#   Positive: 45.36%
#   Negative: 20.00%
#   Neutral: 34.64%
# Community 4:
#   Negative: 37.31%
#   Positive: 39.70%
#   Neutral: 22.99%
# Community 5:
#   Neutral: 49.74%
#   Positive: 25.73%
#   Negative: 24.53%
# Community 6:
#   Negative: 23.92%
#   Positive: 60.61%
#   Neutral: 15.47%
# Community 7:
#   Neutral: 26.53%
#   Negative: 30.69%
#   Positive: 42.77%
# Community 8:
#   Positive: 42.74%
#   Negative: 33.29%
#   Neutral: 23.97%
# Community 9:
#   Negative: 36.81%
#   Positive: 38.54%
#   Neutral: 24.65%
# Community 10:
#   Negative: 25.99%
#   Positive: 50.15%
#   Neutral: 23.85%
# Community 11:
#   Positive: 47.39%
#   Negative: 26.30%
#   Neutral: 26.30%
# Community 12:
#   Negative: 33.46%
#   Positive: 35.69%
#   Neutral: 30.86%
# Community 13:
#   Negative: 27.18%
#   Positive: 39.21%
#   Neutral: 33.61%
# Community 14:
#   Neutral: 30.59%
#   Positive: 48.53%
#   Negative: 20.88%
# Community 15:
#   Negative: 35.36%
#   Positive: 41.99%
#   Neutral: 22.65%
# Community 16:
#   Positive: 47.89%
#   Negative: 17.37%
#   Neutral: 34.74%
# =============================================================================


# Map top bridge nodes (degree centrality) to their communities
top_node_communities = {}
for node, centrality in top_bridge_nodes:
    community = partition[node]
    sentiment = commenter_graph.nodes[node]['sentiment']  # Node sentiment
    if community not in top_node_communities:
        top_node_communities[community] = []
    top_node_communities[community].append((node, centrality, sentiment))

# Print nodes grouped by community
for community, nodes in top_node_communities.items():
    print(f"Community {community}:")
    for node, centrality, sentiment in nodes:
        sentiment_label = {1: "Positive", 0: "Neutral", -1: "Negative"}[sentiment]
        print(f"  Node {node} (Centrality: {centrality:.4f}, Sentiment: {sentiment_label})")
        
# =============================================================================
# Community 1:
#   Node Dillo64 (Centrality: 0.5015, Sentiment: Positive)
# Community 2:
#   Node MariJoyBoy (Centrality: 0.4448, Sentiment: Positive)
# Community 11:
#   Node RemindMeBot (Centrality: 0.4381, Sentiment: Negative)
# Community 6:
#   Node vinsmokewhoswho (Centrality: 0.4357, Sentiment: Positive)
# Community 3:
#   Node Kuro013 (Centrality: 0.3902, Sentiment: Positive)
#   Node Kiosade (Centrality: 0.3794, Sentiment: Neutral)
#   Node ammarbadhrul (Centrality: 0.3611, Sentiment: Positive)
#   Node Dj0sh (Centrality: 0.3329, Sentiment: Negative)
#   Node MaddestChadLad (Centrality: 0.3140, Sentiment: Positive)
# Community 8:
#   Node Asian_Persuasion_1 (Centrality: 0.3893, Sentiment: Positive)
#  

import matplotlib.pyplot as plt
import numpy as np

# Prepare data
labels = [f"Community {i}" for i in range(len(community_sentiments))]
positive = [community_sentiments[i].get(1, 0) for i in range(len(community_sentiments))]
neutral = [community_sentiments[i].get(0, 0) for i in range(len(community_sentiments))]
negative = [community_sentiments[i].get(-1, 0) for i in range(len(community_sentiments))]
bridge_node_counts = [len(top_node_communities.get(i, [])) for i in range(len(community_sentiments))]

x = np.arange(len(labels))
width = 0.4

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [1, 2]})

# Top plot: Top bridge nodes
ax1.bar(x, bridge_node_counts, color='blue', alpha=0.7, label='Top Bridge Node Count')
ax1.set_ylabel('Top Bridge Node Count', fontsize=12)
ax1.set_title('Top Bridge Nodes and Sentiment Distribution by Community', fontsize=16)
ax1.set_xticks(x)
ax1.set_xticklabels([])  # Remove x-axis labels for the top plot
ax1.legend()

# Bottom plot: Sentiment distribution
ax2.bar(x - width, positive, width, label='Positive Sentiment', color='green', alpha=0.7)
ax2.bar(x, neutral, width, label='Neutral Sentiment', color='gray', alpha=0.7)
ax2.bar(x + width, negative, width, label='Negative Sentiment', color='red', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45)
ax2.set_ylabel('Sentiment Proportion', fontsize=12)
ax2.legend()

# Adjust layout
plt.tight_layout()
plt.show()




# =============================================================================
# 
# =============================================================================
#the following doesnt seem needed but I leave it here
# =============================================================================
# # Filter top commenters' activity in high-centrality posts
# top_commenters_high_centrality = top_commenters_data[
#     top_commenters_data['post_id'].isin(high_centrality_post_nodes_revised)
# ]
# 
# # Reset index of commenter_activity to make 'author' a column
# commenter_activity = commenter_activity.reset_index()
# 
# # Group by commenter to count unique posts and total comments
# engagement_summary = (
#     top_commenters_high_centrality.groupby('author')
#     .agg(unique_posts=('post_id', 'nunique'), total_comments=('post_id', 'size'))
#     .reset_index()
# )
# 
# # Merge with total activity to get overall context
# engagement_summary = engagement_summary.merge(
#     commenter_activity[['author', 'total_comments']], on='author', how='left'
# )
# 
# # Calculate the proportion of activity in high-centrality posts
# engagement_summary['proportion_high_centrality'] = (
#     engagement_summary['total_comments_x'] / engagement_summary['total_comments_y']
# )
# 
# # Print summary
# print("Engagement Summary of Top Commenters in High-Centrality Posts:")
# print(engagement_summary)
# 
# =============================================================================





# =============================================================================
# 
# import community as community_louvain  # Install with pip install python-louvain
# 
# # Detect communities using the Louvain algorithm
# partition = community_louvain.best_partition(bi_graph_filtered_data, weight='weight')
# 
# # Assign colors based on communities
# community_colors = {node: f"C{comm}" for node, comm in partition.items()}
# 
# # Spring layout with fixed positions (to preserve the structure)
# pos = nx.spring_layout(bi_graph_filtered_data, seed=42)
# 
# # Plot
# plt.figure(figsize=(12, 12))
# 
# # Draw nodes with community-based coloring
# nx.draw_networkx_nodes(
#     bi_graph_filtered_data,
#     pos,
#     node_color=[community_colors[node] for node in bi_graph_filtered_data.nodes()],
#     node_size=50
# )
# 
# # Draw edges
# nx.draw_networkx_edges(
#     bi_graph_filtered_data, 
#     pos, 
#     alpha=0.3, 
#     edge_color='gray'
# )
# 
# # Add title and display
# plt.title("Communities Detected with Louvain Algorithm", fontsize=14)
# plt.axis('off')
# plt.show()
# 
# # Perform community detection (reuse the code from before or modify as needed)
# import community as community_louvain
# 
# # Detect communities
# partition = community_louvain.best_partition(bi_graph_filtered_data)
# 
# # Assign communities to nodes in the graph
# for node, community in partition.items():
#     bi_graph_filtered_data.nodes[node]['community'] = community
# 
# 
# #check if graph are correctly attached to communities
# print(bi_graph_filtered_data.nodes(data=True))
# 
# 
# #Here I had a mismatch when doing the community algorithm
# #and didnt add the weight to edges it seems, so I had to do it again manually and later on check
# 
# # Assign weights (sentiments) to edges when adding them
# for _, post_data in filtered_posts.iterrows():
#     post_id = post_data['post_id']
#     post_author = post_data['author_x']
# 
#     # Add post as a node
#     bi_graph_filtered_data.add_node(post_id, bipartite=0, author=post_author)
# 
#     # Get commenters for the current post
#     post_comments = filtered_comments[filtered_comments['post_id'] == post_id]
#     for _, comment_data in post_comments.iterrows():
#         commenter = comment_data['author']
#         sentiment = comment_data['sentiment_body']  # Ensure this column contains the sentiment
# 
#         if commenter != '[deleted]':  # Skip deleted users
#             bi_graph_filtered_data.add_node(commenter, bipartite=1)
#             # Add the edge with sentiment as weight
#             bi_graph_filtered_data.add_edge(post_id, commenter, weight=sentiment)
#             
# for u, v, data in list(bi_graph_filtered_data.edges(data=True))[:10]:
#     print(f"Edge ({u}, {v}): Weight = {data.get('weight', None)}")
# 
# 
# # Initialize a dictionary to store average sentiment per community
# community_sentiments = {}
# 
# # Iterate through communities
# for community_id in set(nx.get_node_attributes(bi_graph_filtered_data, 'community').values()):
#     # Find all edges within the community
#     community_edges = [
#         data['weight'] for u, v, data in bi_graph_filtered_data.edges(data=True)
#         if bi_graph_filtered_data.nodes[u]['community'] == community_id and
#            bi_graph_filtered_data.nodes[v]['community'] == community_id
#     ]
# 
#     # Calculate average sentiment
#     if community_edges:  # Avoid division by zero
#         avg_sentiment = sum(community_edges) / len(community_edges)
#     else:
#         avg_sentiment = 0  # No intra-community edges
# 
#     community_sentiments[community_id] = avg_sentiment
# 
# # Print results
# for community_id, avg_sentiment in community_sentiments.items():
#     print(f"Community {community_id}: Average Sentiment = {avg_sentiment}")
#  
#     
# #retrieve the communities after we already launched the community detection algorithm 
# import community as community_louvain  # or use another method depending on what you used
# 
# # Detect communities (assuming the graph is named `bi_graph_filtered_data`)
# partition = community_louvain.best_partition(bi_graph_filtered_data)
# 
# # `partition` is a dictionary: node -> community_id
# 
# #now we create a dictionary to store the communities values we just extracted
# 
# # Initialize an empty dictionary to store communities
# communities = {}
# 
# # Fill the communities dictionary
# for node, community_id in partition.items():
#     if community_id not in communities:
#         communities[community_id] = []
#     communities[community_id].append(node)
# 
# =============================================================================


















