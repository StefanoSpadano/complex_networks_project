# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:21:07 2025

@author: Raffaele
"""

import pandas as pd
import networkx as nx
import numpy as np

from utils import save_plot, compute_flow_values


def load_csv(filepath, **kwargs):
    """
    Load a CSV file into a pandas DataFrame with comprehensive error handling.
    
    Args:
        filepath (str): Path to the CSV file
        **kwargs: Additional arguments to pass to pd.read_csv()
        
    Returns:
        pd.DataFrame or None: Loaded DataFrame or None if loading failed
        
    Raises:
        SystemExit: If file loading fails critically
    """
    try:
        return pd.read_csv(filepath, **kwargs)
    except FileNotFoundError:
        print(f"Error: File not found - {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty - {filepath}")
        return None
    except pd.errors.ParserError:
        print(f"Error: Parsing issue in file - {filepath}")
        return None


def preprocess_dataframe(df, required_columns):
    """
    Clean and preprocess a DataFrame by removing NaNs from required columns.
    
    Args:
        df (pd.DataFrame): DataFrame to preprocess
        required_columns (list): List of column names that cannot have NaNs
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame (modified in place)
    """
    df.dropna(subset=required_columns, inplace=True)
    if 'selftext' in df.columns:
        df['selftext'].fillna('', inplace=True)  # Fill missing post content
    return df


def load_and_preprocess_data():
    """
    Load and preprocess all required datasets for network analysis.
    
    Returns:
        dict: Dictionary containing all loaded and preprocessed DataFrames
        
    Raises:
        SystemExit: If critical data loading fails
    """
    # Define file paths and their configurations
    data_files = {
        'posts': {
            'path': "../data/onepiece_posts.csv",
            'engine': "python",
            'required_columns': ['author', 'created_utc', 'score']
        },
        'comments': {
            'path': "../data/onepiece_comments.csv", 
            'engine': "python",
            'required_columns': ['author', 'created_utc', 'score']
        },
        'filtered_posts': {
            'path': "../data/onepiece_sentiment_posts_filtered.csv",
            'lineterminator': "\n",
            'required_columns': ['post_id', 'author_x']
        },
        'filtered_comments': {
            'path': "../data/onepiece_sentiment_comments_filtered.csv",
            'lineterminator': "\n", 
            'required_columns': ['post_id', 'author', 'sentiment_body']
        }
    }
    
    # Load all datasets
    datasets = {}
    for name, config in data_files.items():
        path = config.pop('path')
        required_cols = config.pop('required_columns')
        
        # Load dataset
        df = load_csv(path, **config)
        if df is None:
            print(f"Failed to load {name} dataset from {path}")
            raise SystemExit("Critical data loading failed. Exiting script.")
            
        # Preprocess dataset
        df = preprocess_dataframe(df, required_cols)
        datasets[name] = df
        
        print(f"✓ Loaded and preprocessed {name}: {len(df)} rows")
    
    return datasets


def build_post_centric_graph(posts_df, comments_df):
    """
    Build a directed graph where post authors are connected to their commenters.
    
    Args:
        posts_df (pd.DataFrame): DataFrame containing post data
        comments_df (pd.DataFrame): DataFrame containing comment data
        
    Returns:
        nx.DiGraph: Directed graph with post-commenter relationships
    """
    # Initialize directed graph
    graph = nx.DiGraph()
    
    # Build graph with comment counts
    for _, post_data in posts_df.iterrows():
        post_id = post_data.get('post_id')
        post_author = post_data.get('author')
        
        if not post_id or not post_author:
            continue  # Skip invalid data
        
        # Add post author node
        graph.add_node(post_author, type='post_author', comment_count=0)
        
        # Get all comments for this post
        post_comments = comments_df[comments_df['post_id'] == post_id]
        
        # Add commenters and edges
        for commenter in post_comments['author']:
            if commenter == '[deleted]':
                continue  # Skip deleted users
            
            # Add or update commenter node
            if commenter not in graph:
                graph.add_node(commenter, type='commenter', comment_count=1)
            else:
                graph.nodes[commenter]['comment_count'] += 1
            
            # Add edge from post author to commenter
            graph.add_edge(post_author, commenter, weight=1)
    
    return graph


def compute_graph_metrics(graph):
    """
    Compute comprehensive metrics for a given graph.
    
    Args:
        graph (nx.Graph or nx.DiGraph): Input graph
        
    Returns:
        dict: Dictionary containing various graph metrics and centrality measures
    """
    metrics = {}
    
    # Basic graph metrics
    metrics['num_nodes'] = graph.number_of_nodes()
    metrics['num_edges'] = graph.number_of_edges()
    
    # Degree metrics
    degrees = dict(graph.degree())
    if degrees:
        max_degree_node = max(degrees, key=degrees.get)
        metrics['max_degree_node'] = max_degree_node
        metrics['max_degree_value'] = degrees[max_degree_node]
    
    # Centrality measures
    try:
        metrics['centrality'] = {
            'degree': nx.degree_centrality(graph),
            'betweenness': nx.betweenness_centrality(graph),
            'closeness': nx.closeness_centrality(graph)
        }
        
        # Create centrality DataFrame for analysis
        centrality_df = pd.DataFrame(metrics['centrality'])
        metrics['centrality_df'] = centrality_df
        
        # Degree centrality statistics
        degree_centralities = list(metrics['centrality']['degree'].values())
        metrics['degree_stats'] = {
            'mean': np.mean(degree_centralities),
            'median': np.median(degree_centralities), 
            'variance': np.var(degree_centralities)
        }
        
    except Exception as e:
        print(f"Warning: Could not compute centrality measures - {e}")
        metrics['centrality'] = None
    
    return metrics


def print_graph_analysis(metrics):
    """
    Print comprehensive analysis of graph metrics.
    
    Args:
        metrics (dict): Dictionary containing graph metrics from compute_graph_metrics()
    """
    print("=== GRAPH ANALYSIS ===")
    print(f"Number of nodes: {metrics['num_nodes']}")
    print(f"Number of edges: {metrics['num_edges']}")
    
    if 'max_degree_node' in metrics:
        print(f"Node with highest degree: {metrics['max_degree_node']} "
              f"({metrics['max_degree_value']} connections)")
    
    if metrics.get('centrality'):
        print("\n=== CENTRALITY ANALYSIS ===")
        centrality_df = metrics['centrality_df']
        
        print("\nTop 10 by Degree Centrality:")
        print(centrality_df.sort_values(by='degree', ascending=False).head(10))
        
        print("\nTop 10 by Betweenness Centrality:")
        print(centrality_df.sort_values(by='betweenness', ascending=False).head(10))
        
        print("\nTop 10 by Closeness Centrality:")
        print(centrality_df.sort_values(by='closeness', ascending=False).head(10))
        
        # Degree centrality statistics
        stats = metrics['degree_stats']
        print(f"\nDegree Centrality Statistics:")
        print(f"Mean: {stats['mean']:.4f}, Median: {stats['median']:.4f}, "
              f"Variance: {stats['variance']:.4f}")
    
    print("=" * 50)
    

def build_bipartite_graph(filtered_posts, filtered_comments):
    """
    Build a bipartite graph where posts and commenters are distinct node types.
    
    Args:
        filtered_posts (pd.DataFrame): DataFrame with filtered posts
        filtered_comments (pd.DataFrame): DataFrame with filtered comments
        
    Returns:
        nx.Graph: Bipartite graph with posts and commenters
    """
    bi_graph = nx.Graph()
    
    # Add post nodes and commenter connections
    for _, post_data in filtered_posts.iterrows():
        post_id = post_data['post_id']
        post_author = post_data['author_x']
        
        # Add post node
        bi_graph.add_node(
            post_id,
            bipartite=0,
            author=post_author,
            type='post'
        )
        
        # Get commenters for this post
        post_comments = filtered_comments[filtered_comments['post_id'] == post_id]
        
        for commenter in post_comments['author']:
            if commenter != '[deleted]':
                # Add commenter node
                bi_graph.add_node(
                    commenter,
                    bipartite=1,
                    type='commenter'
                )
                # Connect post to commenter
                bi_graph.add_edge(post_id, commenter)
    
    return bi_graph


def visualize_bipartite_graph(bi_graph, title="Bipartite Graph Layout"):
    """
    Visualize bipartite graph with posts and commenters in different colors.
    
    Args:
        bi_graph (nx.Graph): The bipartite graph
        title (str): Plot title
        
    Returns:
        dict: Layout positions for reuse
    """
    import matplotlib.pyplot as plt
    
    # Generate layout
    pos = nx.spring_layout(bi_graph)
    
    # Separate node types
    post_nodes = [n for n, d in bi_graph.nodes(data=True) if d['type'] == 'post']
    commenter_nodes = [n for n, d in bi_graph.nodes(data=True) if d['type'] == 'commenter']
    
    # Create visualization
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(bi_graph, pos, nodelist=post_nodes, 
                          node_color='red', node_size=100, label='Posts')
    nx.draw_networkx_nodes(bi_graph, pos, nodelist=commenter_nodes, 
                          node_color='blue', node_size=50, label='Commenters')
    nx.draw_networkx_edges(bi_graph, pos, alpha=0.5)
    
    plt.legend()
    plt.title(title)
    save_plot(title, "plots/network_aspects_plots")
    plt.show()
    
    return pos    

def visualize_bipartite_with_sentiment(bi_graph, filtered_comments, pos, 
                                       title="Bipartite Graph with Sentiment Weights"):
    """
    Visualize bipartite graph with edges colored by sentiment values.
    
    Args:
        bi_graph (nx.Graph): The bipartite graph
        filtered_comments (pd.DataFrame): Comments data with sentiment info
        pos (dict): Node positions from previous layout
        title (str): Plot title
    """
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 15))
    
    # Prepare edge colors and opacities based on sentiment
    edge_colors = []
    edge_opacities = []
    
    for u, v in bi_graph.edges():
        # Find sentiment for this edge (post_id, commenter)
        sentiment_data = filtered_comments.loc[
            (filtered_comments['post_id'] == u) & (filtered_comments['author'] == v),
            'sentiment_body'
        ]
        
        if sentiment_data.empty:
            continue
        
        sentiment_value = sentiment_data.values[0]
        
        # Color based on sentiment
        if sentiment_value > 0:
            edge_colors.append('green')
        elif sentiment_value < 0:
            edge_colors.append('red')
        else:
            edge_colors.append('gray')
        
        # Opacity based on sentiment strength
        edge_opacities.append(min(1, max(0.1, abs(sentiment_value))))
    
    # Draw nodes
    post_nodes = [n for n, d in bi_graph.nodes(data=True) if d['type'] == 'post']
    commenter_nodes = [n for n, d in bi_graph.nodes(data=True) if d['type'] == 'commenter']
    
    nx.draw_networkx_nodes(bi_graph, pos, nodelist=post_nodes, 
                          node_color='red', node_size=100, label='Posts')
    nx.draw_networkx_nodes(bi_graph, pos, nodelist=commenter_nodes, 
                          node_color='blue', node_size=50, label='Commenters')
    
    # Draw edges with sentiment colors
    for (u, v), color, opacity in zip(bi_graph.edges(), edge_colors, edge_opacities):
        nx.draw_networkx_edges(
            bi_graph,
            pos,
            edgelist=[(u, v)],
            edge_color=color,
            alpha=opacity,
            width=1
        )
    
    plt.legend()
    plt.title(title)
    save_plot(title, "plots/network_aspects_plots")
    plt.show()

def detect_communities_louvain(graph):
    """
    Apply Louvain community detection algorithm to the graph.
    
    Args:
        graph (nx.Graph): Input graph for community detection
        
    Returns:
        dict: Dictionary mapping nodes to their community IDs
    """
    from community import community_louvain
    
    # Apply the Louvain method for community detection
    partition = community_louvain.best_partition(graph, weight='weight')
    
    return partition

def analyze_central_nodes(graph, percentile=90):
    """
    Identify and analyze central nodes based on degree centrality threshold.
    
    Args:
        graph (nx.Graph): Input graph
        percentile (int): Percentile threshold for centrality (default 90)
        
    Returns:
        dict: Dictionary containing central nodes analysis
    """
    # Extract degree centrality values
    degree_centrality = nx.degree_centrality(graph)
    if not degree_centrality:
        return {
            'central_nodes': [],
            'threshold': None,
            'degree_centrality': {}
        }
    degree_values = list(degree_centrality.values())
    threshold = np.percentile(degree_values, percentile)
    
    # Filter nodes based on threshold
    central_nodes = [node for node, centrality in degree_centrality.items() 
                    if centrality >= threshold]
    
    return {
        'central_nodes': central_nodes,
        'threshold': threshold,
        'degree_centrality': degree_centrality
    }

def visualize_central_nodes(graph, percentile=90, layout=None, title="Graph Highlighting Central Nodes"):
    """
    Visualize the graph with central nodes highlighted based on degree centrality.
    
    Args:
        graph (nx.Graph): The bipartite graph
        percentile (int): Threshold for central nodes (default is 90th percentile)
        layout (dict): Optional layout positions for reproducibility
        title (str): Title of the plot
    """
    import matplotlib.pyplot as plt

    # Compute degree centrality and threshold
    degree_centrality = nx.degree_centrality(graph)
    degree_values = list(degree_centrality.values())
    threshold = np.percentile(degree_values, percentile)
    central_nodes = [node for node, val in degree_centrality.items() if val >= threshold]

    # Assign node colors and sizes
    node_colors = ["red" if node in central_nodes else "gray" for node in graph.nodes()]
    node_sizes = [1000 if node in central_nodes else 10 for node in graph.nodes()]

    if layout is None:
        layout = nx.spring_layout(graph, seed=42)

    # Plot
    plt.figure(figsize=(15, 15))
    nx.draw(
        graph,
        layout,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color='gray',
        alpha=0.7,
        with_labels=False
    )
    plt.title(title)
    save_plot(title, "plots/network_aspects_plots")
    plt.show()

def visualize_central_subgraph_with_communities(graph, partition, percentile=90, layout=None, title="Central Nodes and Their Communities"):
    """
    Visualize a subgraph of central nodes with communities colored.
    
    Args:
        graph (nx.Graph): Full bipartite graph
        partition (dict): Node-to-community mapping
        percentile (int): Threshold for selecting central nodes
        layout (dict): Layout positions for consistency
        title (str): Title of the plot
    """
    import matplotlib.pyplot as plt

    # Compute central nodes
    degree_centrality = nx.degree_centrality(graph)
    degree_values = list(degree_centrality.values())
    threshold = np.percentile(degree_values, percentile)
    central_nodes = [n for n, c in degree_centrality.items() if c >= threshold]

    # Subgraph
    subgraph = graph.subgraph(central_nodes)

    # Community color map
    community_colors = [partition[n] for n in subgraph.nodes()]

    if layout is None:
        layout = nx.spring_layout(subgraph, seed=42)

    plt.figure(figsize=(12, 12))
    nx.draw(
        subgraph,
        pos=layout,
        node_color=community_colors,
        node_size=[100 if n in central_nodes else 10 for n in subgraph.nodes()],
        edge_color='gray',
        alpha=0.7,
        with_labels=False
    )
    plt.title(title)
    save_plot(title, "plots/network_aspects_plots")
    plt.show()

def plot_centrality_distributions(graph, central_nodes, title_suffix="Central Nodes"):
    """
    Plot the distributions of betweenness and closeness centrality for given central nodes.
    
    Args:
        graph (nx.Graph): Graph containing the nodes
        central_nodes (list): List of node IDs considered central
        title_suffix (str): Used to customize plot titles
    """
    import matplotlib.pyplot as plt

    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)

    central_betweenness = [betweenness[n] for n in central_nodes if n in betweenness]
    central_closeness = [closeness[n] for n in central_nodes if n in closeness]

    plt.figure(figsize=(14, 6))

    # Betweenness
    plt.subplot(1, 2, 1)
    plt.hist(central_betweenness, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Betweenness Centrality of {title_suffix}')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Frequency')

    # Closeness
    plt.subplot(1, 2, 2)
    plt.hist(central_closeness, bins=30, color='salmon', edgecolor='black')
    plt.title(f'Closeness Centrality of {title_suffix}')
    plt.xlabel('Closeness Centrality')
    plt.ylabel('Frequency')

    plt.tight_layout()
    save_plot(f"{title_suffix} Centrality Distributions", "plots/network_aspects_plots")
    plt.show()

def plot_comment_distributions(graph, central_nodes, title_suffix="High Centrality"):
    """
    Plot distributions of comment counts for high centrality posts and commenters.
    
    Args:
        graph (nx.Graph): The bipartite graph
        central_nodes (list): List of central nodes
        title_suffix (str): Title prefix for the plots
    """
    import matplotlib.pyplot as plt

    post_comment_counts = {}
    commenter_comment_counts = {}

    for node in graph.nodes():
        node_type = graph.nodes[node].get("bipartite")
        degree = graph.degree(node)
        if node_type == 0:
            post_comment_counts[node] = degree
        elif node_type == 1:
            commenter_comment_counts[node] = degree

    # Extract only central nodes' values
    high_centrality_posts = [post_comment_counts[n] for n in central_nodes if n in post_comment_counts]
    high_centrality_commenters = [commenter_comment_counts[n] for n in central_nodes if n in commenter_comment_counts]

    # Plot distributions
    plt.figure(figsize=(14, 6))

    # Post comments
    plt.subplot(1, 2, 1)
    plt.hist(high_centrality_posts, bins=30, color='blue', edgecolor='black')
    plt.title(f'Distribution of Comments on {title_suffix} Posts')
    plt.xlabel('Number of Comments')
    plt.ylabel('Frequency')

    # Commenter activity
    plt.subplot(1, 2, 2)
    plt.hist(high_centrality_commenters, bins=30, color='green', edgecolor='black')
    plt.title(f'Distribution of Comments by {title_suffix} Commenters')
    plt.xlabel('Number of Comments')
    plt.ylabel('Frequency')

    plt.tight_layout()
    save_plot(f"Comment Distributions - {title_suffix}", "plots/network_aspects_plots")
    plt.show()

def analyze_post_engagement(filtered_comments, percentile=90):
    """
    Analyze high-engagement posts and top commenters.
    
    Args:
        filtered_comments (pd.DataFrame): DataFrame with comments and post_id
        percentile (int): Top percentile threshold (default 90)
    
    Returns:
        pd.DataFrame: Summary of top posts with commenter stats
    """
    # Pivot table (rows: authors, columns: post_id, values: comment counts)
    heatmap_data = filtered_comments.pivot_table(
        index='author', columns='post_id', aggfunc='size', fill_value=0
    )

    post_engagement = heatmap_data.sum(axis=0)
    threshold = post_engagement.quantile(percentile / 100)
    high_engagement_posts = post_engagement[post_engagement > threshold].index

    high_engagement_comments = filtered_comments[
        filtered_comments['post_id'].isin(high_engagement_posts)
    ]

    commenter_stats = high_engagement_comments.groupby('post_id').agg(
        unique_commenters=('author', 'nunique'),
        total_comments=('author', 'count'),
        top_commenter=('author', lambda x: x.value_counts().idxmax()),
        top_commenter_count=('author', lambda x: x.value_counts().max())
    ).reset_index()

    return commenter_stats

def plot_comment_heatmap(filtered_comments):
    """
    Plot a heatmap of comment counts per author and post.
    
    Args:
        filtered_comments (pd.DataFrame): DataFrame with author and post_id
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    heatmap_data = filtered_comments.pivot_table(
        index='author', columns='post_id', aggfunc='size', fill_value=0
    )

    plt.figure(figsize=(12, 8))
    vmax = heatmap_data.values.max()
    vmax = min(vmax, 20)  # clamp at 20 if very skewed
    sns.heatmap(heatmap_data, cmap="YlGnBu", cbar=True, vmax=vmax)

    plt.title("Comment Count Heatmap (Author vs Post)")
    plt.xlabel("Post ID")
    plt.ylabel("Author")
    save_plot("Comment Count Heatmap", "plots/network_aspects_plots")
    plt.show()

def get_top_commenters(filtered_comments, top_k=20):
    """
    Automatically extract top commenters by volume.
    
    Args:
        filtered_comments (pd.DataFrame): Comments dataset
        top_k (int): Number of top commenters to return
        
    Returns:
        list: Top commenter usernames
    """
    return (
        filtered_comments['author']
        .value_counts()
        .head(top_k)
        .index
        .tolist()
    )


def build_commenter_network(filtered_comments, top_commenters):
    """
    Build an undirected network connecting top commenters who commented on the same post.
    
    Args:
        filtered_comments (pd.DataFrame): DataFrame of comments
        top_commenters (list): List of usernames to include
    
    Returns:
        nx.Graph: Undirected network of commenters
    """
    network = nx.Graph()
    network.add_nodes_from(top_commenters)
    top_df = filtered_comments[filtered_comments['author'].isin(top_commenters)]

    for post_id in top_df['post_id'].unique():
        commenters = top_df[top_df['post_id'] == post_id]['author'].tolist()
        for i in range(len(commenters)):
            for j in range(i + 1, len(commenters)):
                network.add_edge(commenters[i], commenters[j])
    
    return network

def visualize_largest_cluster(graph, title="Mini Cluster of Commenters"):
    """
    Visualize the largest connected component of the given graph.
    
    Args:
        graph (nx.Graph): A commenter network
        title (str): Title for the plot
    """
    import matplotlib.pyplot as plt

    largest_cluster = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_cluster)

    print(f"Cluster nodes: {subgraph.number_of_nodes()}, edges: {subgraph.number_of_edges()}")
    print(f"Cluster density: {nx.density(subgraph):.4f}")

    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(subgraph, seed=42)
    nx.draw(
        subgraph, pos, with_labels=True,
        node_color="skyblue", edge_color="gray", node_size=500, font_size=10
    )
    plt.title(title)
    save_plot(title, "plots/network_aspects_plots")
    plt.show()

    return subgraph

def compute_mini_cluster_centrality(graph):
    """
    Compute centrality measures (degree, betweenness, closeness) for nodes in the mini-cluster.
    
    Args:
        graph (nx.Graph): Subgraph of the mini-cluster
    
    Returns:
        pd.DataFrame: DataFrame sorted by degree centrality
    """
    degree = nx.degree_centrality(graph)
    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)

    df = pd.DataFrame({
        'degree': degree,
        'betweenness': betweenness,
        'closeness': closeness
    }).sort_values(by='degree', ascending=False)

    print("Centrality measures for the mini-cluster:")
    print(df.head())

    return df

def identify_shared_posts(filtered_comments, filtered_posts, cluster_nodes):
    """
    Identify and display posts commented by more than one mini-cluster member.
    
    Args:
        filtered_comments (pd.DataFrame): All filtered comments
        filtered_posts (pd.DataFrame): Original post metadata (for titles)
        cluster_nodes (set): Authors in the mini-cluster
    
    Returns:
        pd.DataFrame: DataFrame with post_id and title
    """
    shared = filtered_comments[
        filtered_comments['author'].isin(cluster_nodes)
    ].groupby('post_id')['author'].apply(list)

    shared = shared[shared.apply(lambda x: len(set(x)) > 1)]
    shared_post_ids = shared.index

    shared_titles = filtered_posts[filtered_posts['post_id'].isin(shared_post_ids)][
        ['post_id', 'title']
    ]

    print("Posts shared by commenters in the mini-cluster:")
    print(shared)
    print("Titles of shared posts:")
    print(shared_titles)

    return shared_titles

def plot_sentiment_distribution(filtered_comments, cluster_nodes, title="Sentiment Distribution for Mini-Cluster Commenters"):
    """
    Plot a pie chart showing sentiment distribution among mini-cluster commenters.
    
    Args:
        filtered_comments (pd.DataFrame): Comment data with sentiment_category
        cluster_nodes (set): Authors in the mini-cluster
        title (str): Title for the pie chart
    """
    import matplotlib.pyplot as plt

    mini_cluster_comments = filtered_comments[
        filtered_comments['author'].isin(cluster_nodes)
    ]
    
    if 'sentiment_category' not in mini_cluster_comments.columns:
        print("Missing 'sentiment_category' column in filtered_comments.")
        return

    sentiment_counts = mini_cluster_comments['sentiment_category'].value_counts()

    plt.figure(figsize=(8, 8))
    colors = ['#ff9999', '#66b3ff', '#99ff99']  # red, blue, green
    sentiment_counts.plot.pie(
        autopct='%1.1f%%', startangle=90, colors=colors,
        labels=sentiment_counts.index
    )
    plt.title(title)
    plt.ylabel('')
    save_plot(title, "plots/network_aspects_plots")
    plt.show()
    
def assign_initial_sentiments(filtered_comments, graph):
    """
    Assign initial sentiment to each node based on average sentiment_body.
    
    Args:
        filtered_comments (pd.DataFrame): Comments with 'sentiment_body'
        graph (nx.Graph): Commenter-commenter network
    
    Returns:
        dict: Mapping from node to initial sentiment (-1, 0, 1)
    """
    def classify(sentiment):
        if sentiment > 0:
            return 1
        elif sentiment < 0:
            return -1
        else:
            return 0

    avg_sentiment = filtered_comments.groupby('author')['sentiment_body'].mean().apply(classify)
    node_sentiments = {node: avg_sentiment.get(node, 0) for node in graph.nodes()}
    nx.set_node_attributes(graph, node_sentiments, 'sentiment')
    return node_sentiments

def propagate_sentiments(graph, max_steps=10, resistance_prob=0.05, flip_prob=0.1):
    """
    Simulate sentiment propagation over the commenter graph.
    
    Args:
        graph (nx.Graph): Commenter-commenter network with 'sentiment' node attributes
        max_steps (int): Maximum propagation steps
        resistance_prob (float): Probability of a node ignoring change
        flip_prob (float): Probability of a node flipping randomly
    
    Returns:
        list[Counter]: Sentiment distribution at each time step
    """
    import random
    from collections import Counter
    sentiment_over_time = []
    current = nx.get_node_attributes(graph, 'sentiment')
    sentiment_over_time.append(Counter(current.values()))

    for step in range(max_steps):
        new_sentiment = {}
        for node in graph.nodes:
            if random.random() < resistance_prob:
                new_sentiment[node] = graph.nodes[node]['sentiment']
                continue
            if random.random() < flip_prob:
                new_sentiment[node] = random.choice([-1, 0, 1])
                continue

            # Weighted sentiment influence from neighbors
            neighbors = graph.neighbors(node)
            influence = sum(graph.nodes[n]['sentiment'] * graph[node][n].get('weight', 1)
                            for n in neighbors)
            new_sentiment[node] = 1 if influence > 0 else -1 if influence < 0 else 0

        nx.set_node_attributes(graph, new_sentiment, 'sentiment')
        sentiment_over_time.append(Counter(new_sentiment.values()))

        if sentiment_over_time[-1] == sentiment_over_time[-2]:
            print(f"Converged at step {step + 1}")
            break

    return sentiment_over_time

def plot_sentiment_evolution(sentiment_over_time):
    """
    Plot how the sentiment proportions evolve over time.
    
    Args:
        sentiment_over_time (list of Counter): Sentiment distribution over time
    """
    import matplotlib.pyplot as plt

    sentiment_labels = [-1, 0, 1]
    proportions = {s: [] for s in sentiment_labels}

    for step in sentiment_over_time:
        total = sum(step.values())
        for s in sentiment_labels:
            proportions[s].append(step.get(s, 0) / total if total > 0 else 0)

    plt.figure(figsize=(10, 6))
    for s in sentiment_labels:
        label = {1: "Positive", 0: "Neutral", -1: "Negative"}[s]
        plt.plot(proportions[s], label=label)
    plt.xlabel("Time Step")
    plt.ylabel("Proportion")
    plt.title("Sentiment Propagation Over Time")
    plt.legend()
    plt.grid(True)
    save_plot("Sentiment Propagation Over Time", "plots/network_aspects_plots")
    plt.show()

def shift_edge_weights(graph):
    """
    Shift all edge weights in the graph to ensure they are non-negative.
    
    Args:
        graph (nx.Graph): Graph possibly with negative weights
    
    Returns:
        nx.Graph: A deep copy with shifted weights
    """
    import copy
    shifted = copy.deepcopy(graph)
    min_weight = min((d['weight'] for _, _, d in shifted.edges(data=True)), default=0)
    if min_weight < 0:
        for u, v, d in shifted.edges(data=True):
            d['weight'] += abs(min_weight)
    return shifted

def analyze_community_sentiments(graph, partition):
    """
    Compute sentiment distribution for each community in the graph.
    
    Args:
        graph (nx.Graph): Graph with 'sentiment' node attribute
        partition (dict): Node → community ID mapping
    
    Returns:
        dict: Community → Counter of sentiment counts
    """
    from collections import Counter
    sentiments = {c: Counter() for c in set(partition.values())}
    for node, comm in partition.items():
        sentiment = graph.nodes[node].get('sentiment')
        if sentiment is not None:
            sentiments[comm][sentiment] += 1
    return sentiments

def print_community_sentiments(sentiment_counts):
    """
    Pretty-print sentiment distribution per community.
    
    Args:
        sentiment_counts (dict): Output from analyze_community_sentiments()
    """
    for community, dist in sentiment_counts.items():
        total = sum(dist.values())
        print(f"Community {community}:")
        for sentiment, count in dist.items():
            label = {1: "Positive", 0: "Neutral", -1: "Negative"}.get(sentiment, sentiment)
            proportion = count / total if total > 0 else 0
            print(f"  {label}: {count} ({proportion:.2%})")

def visualize_communities(graph, partition, highlight_nodes=None, title="Commenter Network Colored by Communities"):
    """
    Visualize a graph with nodes colored by community and optionally highlight specific nodes.
    
    Args:
        graph (nx.Graph): Graph to visualize
        partition (dict): Node → community ID mapping
        highlight_nodes (set or list, optional): Nodes to visually emphasize
        title (str): Title of the plot
    """
    import matplotlib.pyplot as plt

    # Generate color map
    communities = list(set(partition.values()))
    color_map = plt.cm.get_cmap("rainbow", len(communities))
    community_colors = {comm: color_map(i) for i, comm in enumerate(communities)}
    node_colors = [community_colors[partition[n]] for n in graph.nodes()]

    # Determine node sizes
    if highlight_nodes:
        node_sizes = [100 if n in highlight_nodes else 30 for n in graph.nodes()]
    else:
        node_sizes = 30

    # Draw
    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(
        graph,
        pos=pos,
        node_color=node_colors,
        node_size=node_sizes,
        edge_color='gray',
        with_labels=False,
        alpha=0.7
    )
    plt.title(title, fontsize=16)
    save_plot(title, "plots/network_aspects_plots")
    plt.show()

def analyze_top_influencers(graph, top_n=500):
    """
    Identify top influencers using centrality measures on a subgraph.
    
    Args:
        graph (nx.Graph): Full commenter graph
        top_n (int): Number of top nodes by degree to include in subgraph
    
    Returns:
        dict: Contains top influencers and subgraph
    """
    degree = nx.degree_centrality(graph)
    top_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_ids = [node for node, _ in top_nodes]
    subgraph = graph.subgraph(top_ids)

    print(f"Top Commenters Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

    betweenness = nx.betweenness_centrality(subgraph)
    eigenvector = nx.eigenvector_centrality(subgraph)

    top_degree = top_nodes[:5]
    top_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
    top_eigenvector = sorted(eigenvector.items(), key=lambda x: x[1], reverse=True)[:5]

    print("Top Influencers by Degree:", top_degree)
    print("Top Influencers by Betweenness:", top_betweenness)
    print("Top Influencers by Eigenvector:", top_eigenvector)

    return {
        "degree": top_degree,
        "betweenness": top_betweenness,
        "eigenvector": top_eigenvector,
        "subgraph": subgraph
    }

def calculate_sentiment_influence(subgraph):
    """
    Compute a basic sentiment influence score for each node.
    A node scores +1 for each neighbor sharing its sentiment.
    
    Args:
        subgraph (nx.Graph): Subgraph of influencers with sentiment attributes
    
    Returns:
        dict: node → influence score
    """
    scores = {node: 0 for node in subgraph.nodes()}
    for node in subgraph.nodes():
        own_sentiment = subgraph.nodes[node].get('sentiment')
        for neighbor in subgraph.neighbors(node):
            if subgraph.nodes[neighbor].get('sentiment') == own_sentiment:
                scores[node] += 1
    return scores

def compute_modularity(graph, partition):
    """
    Compute the modularity score of a graph given a partition.
    
    Args:
        graph (nx.Graph): The network
        partition (dict): Node → community ID
    
    Returns:
        float: Modularity score
    """
    from community import community_louvain
    modularity = community_louvain.modularity(partition, graph)
    print(f"Modularity Score: {modularity:.4f}")
    return modularity


def compute_sentiment_flow_matrix(graph, partition):
    """
    Create a matrix of sentiment-aligned flows between communities.
    
    Args:
        graph (nx.Graph): Network with node sentiments
        partition (dict): Node → community ID
    
    Returns:
        np.ndarray: Matrix [i][j] = flow from community i to j
    """
    num_comms = len(set(partition.values()))
    matrix = np.zeros((num_comms, num_comms))

    for u, v, d in graph.edges(data=True):
        cu, cv = partition[u], partition[v]
        su = graph.nodes[u].get("sentiment")
        sv = graph.nodes[v].get("sentiment")
        weight = d.get("weight", 1)
        
        if su is not None and sv is not None and su == sv:
           matrix[cu][cv] += weight
           if cu != cv:
            matrix[cv][cu] += weight  # this line adds symmetry

    return matrix


def visualize_sentiment_flow(matrix, title="Sentiment Flow Matrix Heatmap"):
    """
    Plot the sentiment flow matrix using seaborn.
    
    Args:
        matrix (np.ndarray): Sentiment flow matrix
        title (str): Plot title
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="coolwarm")
    plt.title(title)
    plt.xlabel("Community")
    plt.ylabel("Community")
    save_plot(title, "plots/network_aspects_plots")
    plt.show()



def main():
    
    # Sostituisce tutte le righe originali di caricamento
    datasets = load_and_preprocess_data()
    posts_df = datasets['posts']
    comments_df = datasets['comments'] 
    filtered_posts = datasets['filtered_posts']
    filtered_comments = datasets['filtered_comments']
    
    # Sostituisce tutto il blocco di costruzione grafo + analisi
    post_centric_graph = build_post_centric_graph(posts_df, comments_df)
    metrics = compute_graph_metrics(post_centric_graph)
    print_graph_analysis(metrics)
    
    # Dopo il blocco esistente, aggiungi:
    bi_graph_filtered_data = build_bipartite_graph(filtered_posts, filtered_comments)
    pos = visualize_bipartite_graph(bi_graph_filtered_data, "Refined Bipartite Graph Layout")
    
    # Visualizza con sentiment
    visualize_bipartite_with_sentiment(bi_graph_filtered_data, filtered_comments, pos,
                                  "Bipartite Graph with Sentiment Weights (Colored Edges)")
    
    
    #Community detection
    detect_communities_louvain(bi_graph_filtered_data)
    analyze_central_nodes(bi_graph_filtered_data)
    
    visualize_central_nodes(bi_graph_filtered_data, layout=pos)

    partition = detect_communities_louvain(bi_graph_filtered_data)
    visualize_central_subgraph_with_communities(bi_graph_filtered_data, partition, layout=pos)

    degree_centrality = nx.degree_centrality(bi_graph_filtered_data)
    threshold = np.percentile(list(degree_centrality.values()), 90)
    central_nodes = [n for n, c in degree_centrality.items() if c >= threshold]

    plot_centrality_distributions(bi_graph_filtered_data, central_nodes)

    plot_comment_distributions(bi_graph_filtered_data, central_nodes)
    
    # Analisi engagement post
    engagement_stats = analyze_post_engagement(filtered_comments)
    print("High Engagement Post Summary:")
    print(engagement_stats.head())

    # Heatmap (opzionale)
    plot_comment_heatmap(filtered_comments)
    
    top_commenters_list = get_top_commenters(filtered_comments, top_k=20)
    commenter_network = build_commenter_network(filtered_comments, top_commenters_list)

    mini_cluster_graph = visualize_largest_cluster(commenter_network)
    
    # 1. Calcolo centralità nel mini-cluster
    compute_mini_cluster_centrality(mini_cluster_graph)

    # 2. Post condivisi tra membri del mini-cluster
    cluster_nodes = set(mini_cluster_graph.nodes())
    identify_shared_posts(filtered_comments, filtered_posts, cluster_nodes)
    
    plot_sentiment_distribution(filtered_comments, cluster_nodes)
    
    # Costruisci grafo commenter-commenter
    commenter_graph = nx.Graph()
    for post_id, group in filtered_comments.groupby('post_id'):
        commenters = group['author'].unique()
        for i in range(len(commenters)):
            for j in range(i + 1, len(commenters)):
                u, v = commenters[i], commenters[j]
                if not commenter_graph.has_edge(u, v):
                    commenter_graph.add_edge(u, v, weight=0)
                commenter_graph[u][v]['weight'] += 1

    # Assegna sentimenti iniziali
    assign_initial_sentiments(filtered_comments, commenter_graph)

    # Propagazione
    sentiment_evolution = propagate_sentiments(commenter_graph, max_steps=50)

   # Visualizzazione evoluzione
    plot_sentiment_evolution(sentiment_evolution)
    
    # Shift pesi per community detection
    shifted_graph = shift_edge_weights(commenter_graph)

    # Louvain community detection
    from community import community_louvain
    partition = community_louvain.best_partition(shifted_graph)

    # Assegna attributo 'community'
    nx.set_node_attributes(commenter_graph, partition, name='community')

    # Analizza distribuzione dei sentimenti per comunità
    sentiment_counts = analyze_community_sentiments(commenter_graph, partition)
    print_community_sentiments(sentiment_counts)
    
    visualize_communities(commenter_graph, partition, title="Commenter Network Colored by Communities")
    
    influencers = analyze_top_influencers(commenter_graph, top_n=500)
    influence_scores = calculate_sentiment_influence(influencers["subgraph"])
    top_sentiment_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top Influencers by Sentiment Influence:", top_sentiment_influencers)

    top_nodes = {node for node, _ in (
        influencers["degree"] + influencers["betweenness"] + influencers["eigenvector"] + top_sentiment_influencers
    )}

    # Highlight influencer nodes in subgraph
    visualize_communities(
        influencers["subgraph"],
        partition,  # Same partition as main graph
        highlight_nodes=top_nodes,
        title="Top Nodes Highlighted by Community"
    )
    
    # Calcolo modularità
    modularity = compute_modularity(commenter_graph, partition)
    print(modularity)

    # Calcolo sentiment flow
    sentiment_flow_matrix = compute_sentiment_flow_matrix(commenter_graph, partition)
    print("Sentiment Flow Matrix:")
    print(sentiment_flow_matrix)

    # Visualizzazione heatmap
    visualize_sentiment_flow(sentiment_flow_matrix)

    # Calcolo flussi inter e intra
    inter_flows, intra_flows = compute_flow_values(sentiment_flow_matrix)

    print("Top Inter-Community Flows:")
    for i, j, flow in inter_flows[:10]:
        print(f"Community {i} ↔ Community {j}: {flow:.2f}")

    print("Top Intra-Community Flows:")
    for i, flow in intra_flows[:10]:
        print(f"Community {i}: {flow:.2f}")

if __name__ == "__main__":
    main()










    
# =============================================================================
# # Load data with error handling
# def load_csv(filepath):
#     """Load a CSV file into a pandas DataFrame with error handling."""
#     try:
#         return pd.read_csv(filepath)
#     except FileNotFoundError:
#         print(f"Error: File not found - {filepath}")
#         return None
#     except pd.errors.EmptyDataError:
#         print(f"Error: File is empty - {filepath}")
#         return None
#     except pd.errors.ParserError:
#         print(f"Error: Parsing issue in file - {filepath}")
#         return None
# 
# posts_df = pd.read_csv("../data/onepiece_posts.csv", engine="python")
# comments_df = pd.read_csv("../data/onepiece_comments.csv", engine="python")
# 
# if posts_df is None or comments_df is None:
#     raise SystemExit("Data loading failed. Exiting script.")
# 
# # Data Preprocessing
# def preprocess_dataframe(df, required_columns):
#     """Clean and preprocess a DataFrame by removing NaNs from required columns."""
#     df.dropna(subset=required_columns, inplace=True)
#     if 'selftext' in df.columns:
#         df['selftext'].fillna('', inplace=True)  # Fill missing post content
# 
# preprocess_dataframe(posts_df, ['author', 'created_utc', 'score'])
# preprocess_dataframe(comments_df, ['author', 'created_utc', 'score'])
# 
# # Initialize Directed Graph
# post_centric_graph = nx.DiGraph()
# 
# # Build Graph with comment counts
# for _, post_data in posts_df.iterrows():
#     post_id = post_data.get('post_id')  # Ensure key exists
#     post_author = post_data.get('author')
#     
#     if not post_id or not post_author:
#         continue  # Skip invalid data
#     
#     post_centric_graph.add_node(post_author, type='post_author', comment_count=0)
#     
#     post_comments = comments_df[comments_df['post_id'] == post_id]
#     
#     for commenter in post_comments['author']:
#         if commenter == '[deleted]':
#             continue  # Skip deleted users
#         
#         if commenter not in post_centric_graph:
#             post_centric_graph.add_node(commenter, type='commenter', comment_count=1)
#         else:
#             post_centric_graph.nodes[commenter]['comment_count'] += 1
#         
#         post_centric_graph.add_edge(post_author, commenter, weight=1)
# 
# # Compute Basic Graph Metrics
# num_nodes = post_centric_graph.number_of_nodes()
# num_edges = post_centric_graph.number_of_edges()
# print(f"Number of nodes: {num_nodes}")
# print(f"Number of edges: {num_edges}")
# 
# # Degree Metrics
# degrees = dict(post_centric_graph.degree())
# if degrees:
#     max_degree_node = max(degrees, key=degrees.get)
#     print(f"Node with highest degree: {max_degree_node} ({degrees[max_degree_node]} connections)")
# 
# # Centrality Measures first attempt
# centrality_measures = {
#     'degree': nx.degree_centrality(post_centric_graph),
#     'betweenness': nx.betweenness_centrality(post_centric_graph),
#     'closeness': nx.closeness_centrality(post_centric_graph)
# }
# 
# # Convert to DataFrame for easier analysis and print the first 10 to check
# centrality_df = pd.DataFrame(centrality_measures)
# print(centrality_df.sort_values(by='degree', ascending=False).head(10))
# print(centrality_df.sort_values(by='betweenness', ascending=False).head(10))
# print(centrality_df.sort_values(by='closeness', ascending=False).head(10))
# 
# # Degree Centrality Statistics
# degree_centralities = list(centrality_measures['degree'].values())
# mean_degree = np.mean(degree_centralities)
# median_degree = np.median(degree_centralities)
# variance_degree = np.var(degree_centralities)
# print(f"Mean: {mean_degree}, Median: {median_degree}, Variance: {variance_degree}")
# 
# import matplotlib.pyplot as plt
# 
# # Load filtered data, adjust path if necessary
# filtered_posts = pd.read_csv("../data/onepiece_sentiment_posts_filtered.csv", lineterminator="\n")  
# filtered_comments = pd.read_csv("../data/onepiece_sentiment_comments_filtered.csv", lineterminator="\n")  
# 
# # Attempt to visualize the network as a bipartite graph to distinguish the post layer from commenter layer
# bi_graph_filtered_data = nx.Graph()
# 
# # Build the bipartite graph
# for _, post_data in filtered_posts.iterrows():
#     post_id = post_data['post_id']
#     post_author = post_data['author_x']
#     
#     bi_graph_filtered_data.add_node(
#         post_id,
#         bipartite=0,
#         author=post_author,
#         type='post'
#     )
#     
#     post_comments = filtered_comments[filtered_comments['post_id'] == post_id]
#     commenters = post_comments['author']
#     
#     for commenter in commenters:
#         if commenter != '[deleted]':
#             bi_graph_filtered_data.add_node(
#                 commenter,
#                 bipartite=1,
#                 type='commenter'
#             )
#             bi_graph_filtered_data.add_edge(post_id, commenter)
# 
# # Save the layout for future reuse
# pos = nx.spring_layout(bi_graph_filtered_data)
# 
# # Plot refined bipartite graph layout
# plt.figure(figsize=(12, 12))
# nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=[n for n, d in bi_graph_filtered_data.nodes(data=True) if d['type'] == 'post'], node_color='red', node_size=100, label='Posts')
# nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=[n for n, d in bi_graph_filtered_data.nodes(data=True) if d['type'] == 'commenter'], node_color='blue', node_size=50, label='Commenters')
# nx.draw_networkx_edges(bi_graph_filtered_data, pos, alpha=0.5)
# plt.legend()
# plt.title("Refined Bipartite Graph Layout")
# save_plot("Refined Bipartite Graph Layout","plots/network_aspects_plots")
# plt.show()
# 
# # Plot bipartite graph with sentiment-based edge coloring
# plt.figure(figsize=(15, 15))
# edge_colors = []
# edge_opacities = []
# 
# for u, v in bi_graph_filtered_data.edges():
#     sentiment = filtered_comments.loc[
#         (filtered_comments['post_id'] == u) & (filtered_comments['author'] == v),
#         'sentiment_body'
#     ]
#     
#     if sentiment.empty:
#         continue
# 
#     sentiment_value = sentiment.values[0]
#     if sentiment_value > 0:
#         edge_colors.append('green')
#     elif sentiment_value < 0:
#         edge_colors.append('red')
#     else:
#         edge_colors.append('gray')
#     
#     edge_opacities.append(min(1, max(0.1, abs(sentiment_value))))
# 
# nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=[n for n, d in bi_graph_filtered_data.nodes(data=True) if d['type'] == 'post'], node_color='red', node_size=100, label='Posts')
# nx.draw_networkx_nodes(bi_graph_filtered_data, pos, nodelist=[n for n, d in bi_graph_filtered_data.nodes(data=True) if d['type'] == 'commenter'], node_color='blue', node_size=50, label='Commenters')
# 
# for (u, v), color, opacity in zip(bi_graph_filtered_data.edges(), edge_colors, edge_opacities):
#     nx.draw_networkx_edges(
#         bi_graph_filtered_data,
#         pos,
#         edgelist=[(u, v)],
#         edge_color=color,
#         alpha=opacity,
#         width=1
#     )
# 
# plt.legend()
# plt.title("Bipartite Graph with Sentiment Weights (Colored Edges)")
# save_plot("Bipartite Graph with Sentiment Weights (Colored Edges)","plots/network_aspects_plots")
# plt.show()
# 
# 
# #Attempt at community detection algorithm
# 
# from community import community_louvain
# 
# # Apply the Louvain method for community detection
# partition = community_louvain.best_partition(bi_graph_filtered_data, weight='weight')
# 
# # Extract degree centrality values
# degree_centrality = nx.degree_centrality(bi_graph_filtered_data)
# degree_values = list(degree_centrality.values())
# # The value for the threshold was chosen to exclude most of the nodes and include only those with higher degree
# threshold = np.percentile(degree_values, 90)
# 
# # Filter nodes based on threshold
# central_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]
# 
# # Create a color map for central nodes to distinguish them from the other nodes
# central_node_colors = ["red" if node in central_nodes else "gray" for node in bi_graph_filtered_data.nodes()]
# central_node_sizes = [1000 if node in central_nodes else 10 for node in bi_graph_filtered_data.nodes()]
# 
# # Use before defined graph layout
# pos = nx.spring_layout(bi_graph_filtered_data)
# 
# # Plot the graph highlighting central nodes
# plt.figure(figsize=(15, 15))
# nx.draw(
#     bi_graph_filtered_data,
#     pos,
#     node_color=central_node_colors,
#     node_size=central_node_sizes,
#     edge_color='gray',
#     alpha=0.7,
#     with_labels=False
# )
# plt.title("Graph Highlighting Central Nodes (90th Percentile Threshold)")
# save_plot("Graph Highlighting Central Nodes (90th Percentile Threshold)","plots/network_aspects_plots")
# plt.show()
# 
# # Extract subgraph for central nodes to show only those belonging to the central zone
# central_subgraph = bi_graph_filtered_data.subgraph(central_nodes)
# central_node_sizes_subgraph = [100 if node in central_nodes else 10 for node in central_subgraph.nodes()]
# subgraph_community_colors = [partition[node] for node in central_subgraph.nodes()]
# 
# # Plot the subgraph with community detection applied
# # Plot the subgraph with community detection applied
# fig, ax = plt.subplots(figsize=(12, 12))
# 
# nx.draw(
#     central_subgraph,
#     pos,
#     node_color=subgraph_community_colors,
#     node_size=central_node_sizes_subgraph,
#     alpha=0.7,
#     with_labels=False,
#     edge_color='gray',
#     ax=ax  # explicitly link to axes
# )
# 
# ax.set_title("Zoomed-In View: Central Nodes and Their Communities")
# 
# # Save using your helper
# save_plot("Zoomed-In View - Central Nodes and Their Communities", "plots/network_aspects_plots")
# 
# plt.show()
# 
# 
# 
# # Compute betweenness and closeness centrality for central nodes
# betweenness_centrality = nx.betweenness_centrality(bi_graph_filtered_data)
# central_betweenness = {node: betweenness_centrality[node] for node in central_nodes}
# 
# closeness_centrality = nx.closeness_centrality(bi_graph_filtered_data)
# central_closeness = {node: closeness_centrality[node] for node in central_nodes}
# 
# # Plot betweenness and closeness centrality distributions
# plt.figure(figsize=(14, 6))
# 
# # Betweenness Centrality Plot
# plt.subplot(1, 2, 1)
# plt.hist(list(central_betweenness.values()), bins=30, color='skyblue', edgecolor='black')
# plt.title('Betweenness Centrality of Central Nodes')
# plt.xlabel('Betweenness Centrality')
# plt.ylabel('Frequency')
# 
# # Closeness Centrality Plot
# plt.subplot(1, 2, 2)
# plt.hist(list(central_closeness.values()), bins=30, color='salmon', edgecolor='black')
# plt.title('Closeness Centrality of Central Nodes')
# plt.xlabel('Closeness Centrality')
# plt.ylabel('Frequency')
# 
# plt.tight_layout()
# save_plot("Closeness Centrality of Central Nodes","plots/network_aspects_plots")
# plt.show()
# 
# 
# # Compute degree centrality
# degree_centrality = nx.degree_centrality(bi_graph_filtered_data)
# degree_values = list(degree_centrality.values())
# threshold = np.percentile(degree_values, 90)
# central_nodes = [node for node, centrality in degree_centrality.items() if centrality >= threshold]
# 
# # Compute comment counts per post and per commenter
# post_comment_counts = {}
# commenter_comment_counts = {}
# 
# for node in bi_graph_filtered_data.nodes():
#     if bi_graph_filtered_data.nodes[node]["bipartite"] == 0:  # Assuming 0 represents posts
#         post_comment_counts[node] = bi_graph_filtered_data.degree(node)
#     else:  # Assuming 1 represents commenters
#         commenter_comment_counts[node] = bi_graph_filtered_data.degree(node)
# 
# # Extract high centrality posts and commenters
# high_centrality_posts = [post_comment_counts[node] for node in central_nodes if node in post_comment_counts]
# high_centrality_commenters = [commenter_comment_counts[node] for node in central_nodes if node in commenter_comment_counts]
# 
# # Plot distributions
# plt.figure(figsize=(14, 6))
# 
# # Plot distribution of comments on high centrality posts
# plt.subplot(1, 2, 1)
# plt.hist(high_centrality_posts, bins=30, color='blue', edgecolor='black')
# plt.title('Distribution of Comments on High Centrality Posts')
# plt.xlabel('Number of Comments')
# plt.ylabel('Frequency')
# 
# # Plot distribution of comments by high centrality commenters
# plt.subplot(1, 2, 2)
# plt.hist(high_centrality_commenters, bins=30, color='green', edgecolor='black')
# plt.title('Distribution of Comments by High Centrality Commenters')
# plt.xlabel('Number of Comments')
# plt.ylabel('Frequency')
# 
# plt.tight_layout()
# save_plot("Distribution of Comments by High Centrality Commenters","plots/network_aspects_plots")
# plt.show()
# 
# 
# # Load data
# filtered_posts = pd.read_csv("../data/onepiece_sentiment_posts_filtered.csv", lineterminator="\n")
# filtered_comments = pd.read_csv("../data/onepiece_sentiment_comments_filtered.csv", lineterminator="\n")
# 
# # Create pivot table for comments count by author and post
# heatmap_data = filtered_comments.pivot_table(index='author', columns='post_id', aggfunc='size', fill_value=0)
# 
# # Calculate post engagement and identify high engagement posts
# #Only those falling inside the top 10% were labeled as the high_engagement_posts
# post_engagement = heatmap_data.sum(axis=0)
# threshold = post_engagement.quantile(0.9)  # Top 10% threshold
# high_engagement_posts = post_engagement[post_engagement > threshold].index
# 
# # Analyze commenters for high engagement posts
# high_engagement_comments = filtered_comments[filtered_comments['post_id'].isin(high_engagement_posts)]
# commenter_analysis = high_engagement_comments.groupby('post_id').agg(
#     unique_commenters=('author', 'nunique'),
#     total_comments=('author', 'count'),
#     top_commenter=('author', lambda x: x.value_counts().idxmax()),
#     top_commenter_count=('author', lambda x: x.value_counts().max())
# ).reset_index()
# 
# # Clean and check top commenters
# filtered_comments['author_clean'] = filtered_comments['author'].str.strip().str.lower()
# top_commenters_list = ['vinsmokewhoswho', 'kidelaleron', 'totally_not_a_reply', 'kerriazes',
#                        'idkdidkkdkdj', 'scaptastic', 'nicentra', 'hinrik96']
# top_commenters_clean = [x.lower() for x in top_commenters_list]
# missing_commenters = [commenter for commenter in top_commenters_clean if commenter not in filtered_comments['author_clean'].values]
# present_commenters = [commenter for commenter in top_commenters_clean if commenter in filtered_comments['author_clean'].values]
# 
# # Build commenter network
# commenter_network = nx.Graph()
# commenter_network.add_nodes_from(present_commenters)
# filtered_top_comments = filtered_comments[filtered_comments['author'].isin(present_commenters)]
# 
# for post_id in filtered_top_comments['post_id'].unique():
#     commenters = filtered_top_comments[filtered_top_comments['post_id'] == post_id]['author'].tolist()
#     for i in range(len(commenters)):
#         for j in range(i + 1, len(commenters)):
#             commenter_network.add_edge(commenters[i], commenters[j])
# 
# # Visualize the full network
# plt.figure(figsize=(12, 12))
# pos = nx.spring_layout(commenter_network, k=0.15, iterations=20)
# nx.draw_networkx_nodes(commenter_network, pos, node_size=500, node_color='skyblue')
# nx.draw_networkx_edges(commenter_network, pos, width=2, alpha=0.5, edge_color='gray')
# nx.draw_networkx_labels(commenter_network, pos, font_size=12, font_weight='bold')
# plt.title("Commenter Network - Top Commenters")
# save_plot("Commenter Network - Top Commenters","plots/network_aspects_plots")
# plt.show()
# 
# # Analyze the largest connected component (mini-cluster)
# largest_cluster = max(nx.connected_components(commenter_network), key=len)
# mini_cluster_graph = commenter_network.subgraph(largest_cluster)
# 
# print(f"Number of nodes in the cluster: {mini_cluster_graph.number_of_nodes()}")
# print(f"Number of edges in the cluster: {mini_cluster_graph.number_of_edges()}")
# print(f"Cluster density: {nx.density(mini_cluster_graph)}")
# 
# # Visualize the mini-cluster of commenters
# plt.figure(figsize=(8, 8))
# pos = nx.spring_layout(mini_cluster_graph, seed=42)
# nx.draw(mini_cluster_graph, pos, with_labels=True, node_color="skyblue", edge_color="gray", node_size=500, font_size=10)
# plt.title("Mini Cluster of Commenters")
# save_plot("Mini Cluster of Commenters","plots/network_aspects_plots")
# plt.show()
# 
# # Centrality analysis applied to the mini cluster of commenters just found; saving them in a dataframe
# centrality_df_mini_cluster = pd.DataFrame({
#     'degree': nx.degree_centrality(mini_cluster_graph),
#     'betweenness': nx.betweenness_centrality(mini_cluster_graph),
#     'closeness': nx.closeness_centrality(mini_cluster_graph)
# }).sort_values(by='degree', ascending=False)
# 
# print("Centrality measures for the mini-cluster:")
# print(centrality_df_mini_cluster)
# 
# # Identify shared posts in the mini-cluster
# shared_posts = filtered_comments[filtered_comments['author'].isin(largest_cluster)].groupby('post_id')['author'].apply(list)
# shared_posts = shared_posts[shared_posts.apply(len) > 1]
# shared_post_ids = shared_posts.index
# shared_post_titles = filtered_posts[filtered_posts['post_id'].isin(shared_post_ids)][['post_id', 'title']]
# 
# print("Posts shared by commenters in the mini-cluster:")
# print(shared_posts)
# print("Titles of shared posts:")
# print(shared_post_titles)
# 
# # Analyze sentiment distribution for mini-cluster commenters and visualize it as a pie chart
# mini_cluster_comments = filtered_comments[filtered_comments['author'].isin(present_commenters)]
# sentiment_counts = mini_cluster_comments['sentiment_category'].value_counts()
# 
# plt.figure(figsize=(8, 8))
# colors = ['#ff9999', '#66b3ff', '#99ff99']
# sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=colors, labels=sentiment_counts.index)
# plt.title("Sentiment Distribution for Mini-Cluster Commenters")
# plt.ylabel('')
# save_plot("Sentiment Distribution for Mini-Cluster Commenters","plots/network_aspects_plots")
# plt.show()
# 
# 
# #Follows an attempt at simulating propagation of sentiments across the network of commenters
# #who commented on the same post
# 
# from collections import Counter
# import random
# 
# 
# # Set random seeds for reproducibility
# random.seed(42)
# np.random.seed(42)
# 
# # Build the Commenter-Commenter Network 
# commenter_graph = nx.Graph()
# 
# # Create edges between commenters who commented on the same post
# for post_id, post_comments in filtered_comments.groupby('post_id'):
#     commenters = post_comments['author'].unique()
#     for i in range(len(commenters)):
#         for j in range(i + 1, len(commenters)):
#             commenter1, commenter2 = commenters[i], commenters[j]
#             if not commenter_graph.has_edge(commenter1, commenter2):
#                 commenter_graph.add_edge(commenter1, commenter2, weight=0)
#             commenter_graph[commenter1][commenter2]['weight'] += 1
# 
# # Assign sentiments based on sentiment_body saved earlier
# 
# def assign_initial_sentiment(row):
#     if row > 0:
#         return 1  # Positive
#     elif row < 0:
#         return -1  # Negative
#     else:
#         return 0  # Neutral
# 
# # Aggregate sentiment_body by author and assign initial sentiments
# # here I used average sentiments for each comments authored by the same commenter
# 
# initial_sentiments = filtered_comments.groupby('author')['sentiment_body'].mean().apply(assign_initial_sentiment)
# nx.set_node_attributes(commenter_graph, initial_sentiments.to_dict(), name='sentiment')
# 
# # Print initial sentiment distribution
# 
# initial_distribution = Counter(nx.get_node_attributes(commenter_graph, 'sentiment').values())
# print("Initial Sentiment Distribution:")
# print(f"Positive: {initial_distribution[1]}")
# print(f"Neutral: {initial_distribution[0]}")
# print(f"Negative: {initial_distribution[-1]}")
# 
# # Simulate sentiment propagation 
# # resistance_prob emulates the stubborness in real_world networks
# # flip_prob allows for adapting to random sentiments mimicking unpredictable external factors
# # a weighted influence from neighbors is the main mechanism for sentiments updates
# 
# def propagate_sentiments(graph, max_steps=10, resistance_prob=0.05, flip_prob=0.1):
#     sentiment_over_time = []
#     current_sentiments = nx.get_node_attributes(graph, 'sentiment')
#     sentiment_over_time.append(Counter(current_sentiments.values()))
# 
#     for step in range(max_steps):
#         new_sentiments = {}
#         for node in graph.nodes:
#             if random.random() < resistance_prob:
#                 new_sentiments[node] = graph.nodes[node]['sentiment']
#                 continue
# 
#             if random.random() < flip_prob:
#                 new_sentiments[node] = random.choice([-1, 0, 1])
#                 continue
# 
#             # Weighted influence from neighbors
#             neighbors = list(graph.neighbors(node))
#             weighted_sum = sum(
#                 graph.nodes[neighbor]['sentiment'] * graph[node][neighbor]['weight']
#                 for neighbor in neighbors
#             )
#             new_sentiments[node] = 1 if weighted_sum > 0 else -1 if weighted_sum < 0 else 0
# 
#         nx.set_node_attributes(graph, new_sentiments, 'sentiment')
#         sentiment_distribution = Counter(new_sentiments.values())
#         sentiment_over_time.append(sentiment_distribution)
# 
#         # Check for convergence
#         if sentiment_over_time[-1] == sentiment_over_time[-2]:
#             print(f"Converged at step {step + 1}")
#             break
# 
#     return sentiment_over_time
# 
# # Backup original sentiments
# 
# original_sentiments = nx.get_node_attributes(commenter_graph, 'sentiment')
# 
# # Run sentiment propagation
# sentiment_evolution = propagate_sentiments(commenter_graph, max_steps=50)
# 
# # Restore original sentiments
# # was used during testing to check for possible different results obtained from the sentiment propagation
# 
# nx.set_node_attributes(commenter_graph, original_sentiments, 'sentiment')
# 
# # Visualize sentiment propagation 
# sentiment_labels = [-1, 0, 1]  # Negative, Neutral, Positive
# proportions = {label: [] for label in sentiment_labels}
# 
# for distribution in sentiment_evolution:
#     total = sum(distribution.values())
#     for label in sentiment_labels:
#         proportions[label].append(distribution.get(label, 0) / total)
# 
# # Plot sentiment evolution over a fixed number of times steps
# 
# plt.figure(figsize=(10, 6))
# for label, values in proportions.items():
#     plt.plot(values, label={-1: "Negative", 0: "Neutral", 1: "Positive"}[label])
# plt.xlabel("Time Step")
# plt.ylabel("Proportion of Sentiments")
# plt.title("Sentiment Propagation Over Time")
# plt.legend()
# plt.grid(True)
# save_plot("Sentiment Propagation Over Time","plots/network_aspects_plots")
# plt.show()
# 
# # Check for Convergence 
# # during the sentiment propagation attempt all the parameters were chosen in heuristic way
# 
# convergence_threshold = 0.001
# converged = True
# 
# for t in range(1, len(sentiment_evolution)):
#     prev_dist = sentiment_evolution[t - 1]
#     curr_dist = sentiment_evolution[t]
#     total_nodes = sum(prev_dist.values())
# 
#     diffs = [
#         abs(curr_dist.get(label, 0) / total_nodes - prev_dist.get(label, 0) / total_nodes)
#         for label in [-1, 0, 1]
#     ]
# 
#     if any(diff > convergence_threshold for diff in diffs):
#         converged = False
#         break
# 
# print("The system has reached convergence." if converged else "The system has NOT converged yet.")
# 
# 
# import copy
# 
# 
# # Shift Edge Weights for Community Detection as the Louvain algorithm needs positive edges only
# 
# def shift_edge_weights(graph):
#     """
#     Shift edge weights to make them non-negative for community detection.
#     
#     Args:
#         graph (nx.Graph): The input graph with potentially negative edge weights.
#     
#     Returns:
#         nx.Graph: A copy of the graph with shifted edge weights.
#     """
#     shifted_graph = copy.deepcopy(graph)
#     min_weight = min(d['weight'] for _, _, d in shifted_graph.edges(data=True))
#     for u, v, d in shifted_graph.edges(data=True):
#         d['weight'] += abs(min_weight)
#     return shifted_graph
# 
# # Shift edge weights and perform community detection
# shifted_graph = shift_edge_weights(commenter_graph)
# partition = community_louvain.best_partition(shifted_graph)
# 
# # Analyze and print community sizes
# community_sizes = Counter(partition.values())
# print("Community Sizes (number of nodes per community):")
# for community, size in community_sizes.items():
#     print(f"Community {community}: {size} nodes")
# 
# # Assign communities as node attributes for further analysis
# nx.set_node_attributes(commenter_graph, partition, name='community')
# 
# # Analyze Sentiment Distribution per Community 
# 
# def analyze_community_sentiments(graph, partition):
#     """
#     Analyze sentiment distribution for each community.
#     
#     Args:
#         graph (nx.Graph): The graph with node sentiments.
#         partition (dict): A dictionary mapping nodes to their communities.
#     
#     Returns:
#         dict: A dictionary mapping communities to their sentiment distributions.
#     """
#     community_sentiments = {c: Counter() for c in set(partition.values())}
#     for node, community in partition.items():
#         sentiment = graph.nodes[node]['sentiment']
#         community_sentiments[community][sentiment] += 1
#     return community_sentiments
# 
# # Populate and print sentiment distribution per community
# 
# community_sentiments = analyze_community_sentiments(commenter_graph, partition)
# print("Sentiment Distribution per Community:")
# for community, sentiments in community_sentiments.items():
#     total = sum(sentiments.values())
#     print(f"Community {community}:")
#     for sentiment, count in sentiments.items():
#         proportion = count / total
#         label = {1: "Positive", 0: "Neutral", -1: "Negative"}[sentiment]
#         print(f"  {label}: {count} ({proportion:.2%})")
# 
# # Visualize Communities 
# 
# def visualize_communities(graph, partition, highlight_nodes=None, title="Commenters Network"):
#     """
#     Visualize the graph with nodes colored by their communities.
#     Optionally highlight specific nodes (e.g., top influencers).
#     
#     Args:
#         graph (nx.Graph): The graph to visualize.
#         partition (dict): A dictionary mapping nodes to their communities.
#         highlight_nodes (set): A set of nodes to highlight (e.g., top influencers).
#         title (str): Title of the plot.
#     """
#     # Assign colors based on communities
#     num_communities = len(set(partition.values()))
#     color_map = plt.cm.rainbow
#     colors = [color_map(i / num_communities) for i in range(num_communities)]
#     community_colors = {community: colors[i] for i, community in enumerate(set(partition.values()))}
#     node_colors = [community_colors[partition[node]] for node in graph.nodes]
# 
#     # Assign node sizes based on highlighting
#     node_sizes = [100 if node in highlight_nodes else 30 for node in graph.nodes] if highlight_nodes else 30
# 
#     # Create the figure and axes
#     plt.figure(figsize=(12, 12))
#     pos = nx.spring_layout(graph, seed=42)  # Use spring_layout for consistency
# 
#     # Draw the graph
#     nx.draw_networkx(
#         graph,
#         pos=pos,
#         with_labels=False,
#         node_size=node_sizes,
#         node_color=node_colors,
#         edge_color='gray',
#         alpha=0.7
#     )
# 
#     plt.title(title, fontsize=16)
#     save_plot(title,"plots/network_aspects_plots")
#     plt.show()
# 
# # Visualize the full graph with communities
# visualize_communities(commenter_graph, partition, title="Commenter Network Colored by Communities")
# 
# #  Analyze Top Influencers (Optimized) 
# # there were some issues calculating betweenness centrality for this amount of nodes
# # as the betweenness centrality takes O(N^3) to O(N^2) steps so for this graph was challenging
# 
# def analyze_top_influencers(graph, top_n=500):
#     """
#     Analyze top influencers in the graph using centrality measures.
#     First, select the top `top_n` nodes by degree centrality.
#     Then, calculate betweenness and eigenvector centrality only for this subset.
#     
#     Args:
#         graph (nx.Graph): The graph to analyze.
#         top_n (int): The number of top nodes to consider.
#     
#     Returns:
#         dict: A dictionary containing top influencers by different centrality measures.
#     """
#     # Step 1: Select top nodes by degree centrality
#     degree_centrality = nx.degree_centrality(graph)
#     top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
#     top_node_ids = [node for node, _ in top_nodes]
# 
#     # Create a subgraph from the top nodes
#     subgraph = graph.subgraph(top_node_ids)
#     print(f"Top Commenters Subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
# 
#     # Step 2: Calculate betweenness centrality for the subgraph
#     betweenness_centrality = nx.betweenness_centrality(subgraph, normalized=True)
#     top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
# 
#     # Step 3: Calculate eigenvector centrality for the subgraph
#     eigenvector_centrality = nx.eigenvector_centrality(subgraph)
#     top_eigenvector = sorted(eigenvector_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
# 
#     # Step 4: Get top degree centrality nodes (already computed)
#     top_degree = top_nodes[:5]
# 
#     print("Top Influencers by Degree Centrality:", top_degree)
#     print("Top Influencers by Betweenness Centrality:", top_betweenness)
#     print("Top Influencers by Eigenvector Centrality:", top_eigenvector)
# 
#     return {
#         "degree": top_degree,
#         "betweenness": top_betweenness,
#         "eigenvector": top_eigenvector,
#         "subgraph": subgraph
#     }
# 
# # Analyze top influencers
# influencers = analyze_top_influencers(commenter_graph, top_n=500)  # Limit to top 500 nodes
# 
# # Calculate Sentiment Influence Scores 
# 
# def calculate_sentiment_influence(subgraph):
#     """
#     Calculate sentiment influence scores for nodes in the subgraph.
#     
#     Args:
#         subgraph (nx.Graph): The subgraph to analyze.
#     
#     Returns:
#         dict: A dictionary mapping nodes to their sentiment influence scores.
#     """
#     influence_scores = {node: 0 for node in subgraph.nodes}
#     for node in subgraph.nodes:
#         for neighbor in subgraph.neighbors(node):
#             if subgraph.nodes[node]['sentiment'] == subgraph.nodes[neighbor]['sentiment']:
#                 influence_scores[node] += 1
#     return influence_scores
# 
# # Get top sentiment influencers
# 
# influence_scores = calculate_sentiment_influence(influencers["subgraph"])
# top_sentiment_influencers = sorted(influence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
# print("Top Influencers by Sentiment Influence:", top_sentiment_influencers)
# 
# # Visualize Subgraph with Top Nodes Highlighted 
# 
# top_nodes = {node for node, _ in influencers["degree"] + influencers["betweenness"] + influencers["eigenvector"] + top_sentiment_influencers}
# visualize_communities(influencers["subgraph"], partition, highlight_nodes=top_nodes, title="Top Nodes Highlighted by Community")
# 
# 
# # Compute Modularity Score 
# # this quantity helps understanding is the communities just found are somewhat meaningful or not
# 
# def compute_modularity(graph, partition):
#     """
#     Compute the modularity score for the given graph and partition.
#     
#     Args:
#         graph (nx.Graph): The input graph.
#         partition (dict): A dictionary mapping nodes to their communities.
#     
#     Returns:
#         float: The modularity score.
#     """
#     modularity = community_louvain.modularity(partition, graph)
#     print(f"Modularity Score: {modularity}")
#     return modularity
# 
# # Compute modularity score
# modularity = compute_modularity(commenter_graph, partition)
# 
# # Compute Sentiment Flow Matrix 
# # this is an attempt to check for propagation across the communities just found with the community detection algorithm
# 
# def compute_sentiment_flow_matrix(graph, partition):
#     """
#     Compute the sentiment flow matrix between communities.
#     
#     Args:
#         graph (nx.Graph): The input graph.
#         partition (dict): A dictionary mapping nodes to their communities.
#     
#     Returns:
#         np.ndarray: A matrix where each cell (i, j) represents the sentiment flow from community i to j.
#     """
#     num_communities = len(set(partition.values()))
#     sentiment_flow_matrix = np.zeros((num_communities, num_communities))
# 
#     # Iterate over edges in the graph
#     for u, v, d in graph.edges(data=True):
#         community_u = partition[u]
#         community_v = partition[v]
#         weight = d.get('weight', 1)  # Default weight to 1 if not set
#         sentiment_u = graph.nodes[u]['sentiment']
#         sentiment_v = graph.nodes[v]['sentiment']
# 
#         # Increment the flow based on sentiment alignment
#         if sentiment_u == sentiment_v:
#             sentiment_flow_matrix[community_u, community_v] += weight
# 
#     return sentiment_flow_matrix
# 
# # Compute sentiment flow matrix
# sentiment_flow_matrix = compute_sentiment_flow_matrix(commenter_graph, partition)
# 
# # Print the sentiment flow matrix
# print("Sentiment Flow Matrix:")
# print(sentiment_flow_matrix)
# 
# import seaborn as sns
# 
# # Visualize Sentiment Flow Matrix 
# def visualize_sentiment_flow(matrix):
#     """
#     Visualize the sentiment flow matrix as a heatmap.
#     
#     Args:
#         matrix (np.ndarray): The sentiment flow matrix.
#     """
#     plt.figure(figsize=(12, 10))
#     sns.heatmap(matrix, annot=True, fmt=".1f", cmap="coolwarm")
#     plt.title("Sentiment Flow Matrix Heatmap")
#     plt.xlabel("Community")
#     plt.ylabel("Community")
#     save_plot("Sentiment Flow Matrix Heatmap","plots/network_aspects_plots")
#     plt.show()
# 
# # Visualize the sentiment flow matrix
# visualize_sentiment_flow(sentiment_flow_matrix)
# 
# # Compute Inter and Intra Flow Values 
# # these values are computed adding sentiment weights if there is an alignment between nodes sentiment
# 
# 
# # Compute inter and intra flow values
# inter_flows, intra_flows = compute_flow_values(sentiment_flow_matrix)
# 
# # Print top inter-community flows
# print("Top Inter-Community Flows:")
# for i, j, flow in inter_flows[:10]:  # Top 10 flows
#     print(f"Community {i} ↔ Community {j}: {flow:.2f}")
# 
# # Print top intra-community flows
# print("Top Intra-Community Flows:")
# for i, flow in intra_flows[:10]:  # Top 10 flows
#     print(f"Community {i}: {flow:.2f}")
# =============================================================================
