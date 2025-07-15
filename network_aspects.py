# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:21:07 2025

@author: Raffaele
"""

import pandas as pd
import networkx as nx
import numpy as np
import configparser
import argparse
import os
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


def load_and_preprocess_data(posts_file, comments_file, filtered_posts_file, filtered_comments_file):
    """
    Load and preprocess all required datasets for network analysis.

    Args:
        posts_file (str): Path to posts CSV
        comments_file (str): Path to comments CSV
        filtered_posts_file (str): Path to filtered posts CSV
        filtered_comments_file (str): Path to filtered comments CSV

    Returns:
        dict: Dictionary containing all loaded and preprocessed DataFrames

    Raises:
        SystemExit: If critical data loading fails
    """
    data_files = {
        'posts': {
            'path': posts_file,
            'engine': "python",
            'required_columns': ['author', 'created_utc', 'score']
        },
        'comments': {
            'path': comments_file,
            'engine': "python",
            'required_columns': ['author', 'created_utc', 'score']
        },
        'filtered_posts': {
            'path': filtered_posts_file,
            'lineterminator': "\n",
            'required_columns': ['post_id', 'author_x']
        },
        'filtered_comments': {
            'path': filtered_comments_file,
            'lineterminator': "\n",
            'required_columns': ['post_id', 'author', 'sentiment_body']
        }
    }

    datasets = {}
    for name, config in data_files.items():
        path = config.pop('path')
        required_cols = config.pop('required_columns')

        df = load_csv(path, **config)
        if df is None:
            print(f" Failed to load {name} dataset from {path}")
            raise SystemExit("Critical data loading failed. Exiting script.")

        df = preprocess_dataframe(df, required_cols)
        datasets[name] = df

        print(f" Loaded and preprocessed {name}: {len(df)} rows")

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
    #Initialize directed graph
    graph = nx.DiGraph()
    
    #Build graph with comment counts
    for _, post_data in posts_df.iterrows():
        post_id = post_data.get('post_id')
        post_author = post_data.get('author')
        
        if not post_id or not post_author:
            continue  #Skip invalid data
        
        #Add post author node
        graph.add_node(post_author, type='post_author', comment_count=0)
        
        #Get all comments for this post
        post_comments = comments_df[comments_df['post_id'] == post_id]
        
        #Add commenters and edges
        for commenter in post_comments['author']:
            if commenter == '[deleted]':
                continue  #Skip deleted users
            
            #Add or update commenter node
            if commenter not in graph:
                graph.add_node(commenter, type='commenter', comment_count=1)
            else:
                graph.nodes[commenter]['comment_count'] += 1
            
            #Add edge from post author to commenter
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
    
    #Basic graph metrics
    metrics['num_nodes'] = graph.number_of_nodes()
    metrics['num_edges'] = graph.number_of_edges()
    
    #Degree metrics
    degrees = dict(graph.degree())
    if degrees:
        max_degree_node = max(degrees, key=degrees.get)
        metrics['max_degree_node'] = max_degree_node
        metrics['max_degree_value'] = degrees[max_degree_node]
    
    #Centrality measures
    try:
        metrics['centrality'] = {
            'degree': nx.degree_centrality(graph),
            'betweenness': nx.betweenness_centrality(graph),
            'closeness': nx.closeness_centrality(graph)
        }
        
        #Create centrality DataFrame for analysis
        centrality_df = pd.DataFrame(metrics['centrality'])
        metrics['centrality_df'] = centrality_df
        
        #Degree centrality statistics
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
    
    #Add post nodes and commenter connections
    for _, post_data in filtered_posts.iterrows():
        post_id = post_data['post_id']
        post_author = post_data['author_x']
        
        #Add post node
        bi_graph.add_node(
            post_id,
            bipartite=0,
            author=post_author,
            type='post'
        )
        
        #Get commenters for this post
        post_comments = filtered_comments[filtered_comments['post_id'] == post_id]
        
        for commenter in post_comments['author']:
            if commenter != '[deleted]':
                #Add commenter node
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
    
    #Generate layout
    pos = nx.spring_layout(bi_graph)
    
    #Separate node types
    post_nodes = [n for n, d in bi_graph.nodes(data=True) if d['type'] == 'post']
    commenter_nodes = [n for n, d in bi_graph.nodes(data=True) if d['type'] == 'commenter']
    
    #Create visualization
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
    
    #Prepare edge colors and opacities based on sentiment
    edge_colors = []
    edge_opacities = []
    
    for u, v in bi_graph.edges():
        #Find sentiment for this edge (post_id, commenter)
        sentiment_data = filtered_comments.loc[
            (filtered_comments['post_id'] == u) & (filtered_comments['author'] == v),
            'sentiment_body'
        ]
        
        if sentiment_data.empty:
            continue
        
        sentiment_value = sentiment_data.values[0]
        
        #Color based on sentiment
        if sentiment_value > 0:
            edge_colors.append('green')
        elif sentiment_value < 0:
            edge_colors.append('red')
        else:
            edge_colors.append('gray')
        
        #Opacity based on sentiment strength
        edge_opacities.append(min(1, max(0.1, abs(sentiment_value))))
    
    #Draw nodes
    post_nodes = [n for n, d in bi_graph.nodes(data=True) if d['type'] == 'post']
    commenter_nodes = [n for n, d in bi_graph.nodes(data=True) if d['type'] == 'commenter']
    
    nx.draw_networkx_nodes(bi_graph, pos, nodelist=post_nodes, 
                          node_color='red', node_size=100, label='Posts')
    nx.draw_networkx_nodes(bi_graph, pos, nodelist=commenter_nodes, 
                          node_color='blue', node_size=50, label='Commenters')
    
    #Draw edges with sentiment colors
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
    
    #Apply the Louvain method for community detection
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
    #Extract degree centrality values
    degree_centrality = nx.degree_centrality(graph)
    if not degree_centrality:
        return {
            'central_nodes': [],
            'threshold': None,
            'degree_centrality': {}
        }
    degree_values = list(degree_centrality.values())
    threshold = np.percentile(degree_values, percentile)
    
    #Filter nodes based on threshold
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

    #Compute degree centrality and threshold
    degree_centrality = nx.degree_centrality(graph)
    degree_values = list(degree_centrality.values())
    
    if not degree_values:
        print("Graph is empty. Skipping centrality analysis.")
        return  #Exit this function early

    threshold = np.percentile(degree_values, percentile)
    central_nodes = [node for node, val in degree_centrality.items() if val >= threshold]

    #Assign node colors and sizes
    node_colors = ["red" if node in central_nodes else "gray" for node in graph.nodes()]
    node_sizes = [1000 if node in central_nodes else 10 for node in graph.nodes()]

    if layout is None:
        layout = nx.spring_layout(graph, seed=42)

    #Plot
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

    #Compute central nodes
    degree_centrality = nx.degree_centrality(graph)
    degree_values = list(degree_centrality.values())
    threshold = np.percentile(degree_values, percentile)
    central_nodes = [n for n, c in degree_centrality.items() if c >= threshold]

    #Subgraph
    subgraph = graph.subgraph(central_nodes)

    #Community color map
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

    #Betweenness
    plt.subplot(1, 2, 1)
    plt.hist(central_betweenness, bins=30, color='skyblue', edgecolor='black')
    plt.title(f'Betweenness Centrality of {title_suffix}')
    plt.xlabel('Betweenness Centrality')
    plt.ylabel('Frequency')

    #Closeness
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

    #Extract only central nodes' values
    high_centrality_posts = [post_comment_counts[n] for n in central_nodes if n in post_comment_counts]
    high_centrality_commenters = [commenter_comment_counts[n] for n in central_nodes if n in commenter_comment_counts]

    #Plot distributions
    plt.figure(figsize=(14, 6))

    #Post comments
    plt.subplot(1, 2, 1)
    plt.hist(high_centrality_posts, bins=30, color='blue', edgecolor='black')
    plt.title(f'Distribution of Comments on {title_suffix} Posts')
    plt.xlabel('Number of Comments')
    plt.ylabel('Frequency')

    #Commenter activity
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
        filtered_comments (pd.DataFrame): DataFrame with 'author' and 'post_id'
        percentile (int): Top percentile threshold

    Returns:
        pd.DataFrame: Summary DataFrame with post_id and engagement stats
    """
    if filtered_comments.empty or "author" not in filtered_comments.columns or "post_id" not in filtered_comments.columns:
        return pd.DataFrame(columns=[
            "post_id", "unique_commenters", "total_comments", "top_commenter", "top_commenter_count"
        ])

    heatmap_data = filtered_comments.pivot_table(
        index='author', columns='post_id', aggfunc='size', fill_value=0
    )

    post_engagement = heatmap_data.sum(axis=0)
    if post_engagement.empty:
        return pd.DataFrame(columns=[
            "post_id", "unique_commenters", "total_comments", "top_commenter", "top_commenter_count"
        ])

    threshold = post_engagement.quantile(percentile / 100)
    high_engagement_posts = post_engagement[post_engagement >= threshold].index

    high_engagement_comments = filtered_comments[
        filtered_comments['post_id'].isin(high_engagement_posts)
    ]

    if high_engagement_comments.empty:
        return pd.DataFrame(columns=[
            "post_id", "unique_commenters", "total_comments", "top_commenter", "top_commenter_count"
        ])

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
    Build an undirected weighted network connecting top commenters who commented on the same post.
    
    Args:
        filtered_comments (pd.DataFrame): DataFrame of comments
        top_commenters (list): List of usernames to include
    
    Returns:
        nx.Graph: Undirected network of commenters with edge weights as shared post counts
    """
    network = nx.Graph()
    network.add_nodes_from(top_commenters)
    top_df = filtered_comments[filtered_comments['author'].isin(top_commenters)]

    for post_id in top_df['post_id'].unique():
        commenters = top_df[top_df['post_id'] == post_id]['author'].tolist()
        for i in range(len(commenters)):
            for j in range(i + 1, len(commenters)):
                u, v = commenters[i], commenters[j]
                if network.has_edge(u, v):
                    network[u][v]["weight"] += 1
                else:
                    network.add_edge(u, v, weight=1)
    
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

    #Generate color map
    communities = list(set(partition.values()))
    color_map = plt.cm.get_cmap("rainbow", len(communities))
    community_colors = {comm: color_map(i) for i, comm in enumerate(communities)}
    node_colors = [community_colors[partition[n]] for n in graph.nodes()]

    #Determine node sizes
    if highlight_nodes:
        node_sizes = [100 if n in highlight_nodes else 30 for n in graph.nodes()]
    else:
        node_sizes = 30

    #Draw
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
    eigenvector = nx.eigenvector_centrality(subgraph, max_iter=1000, tol=1e-06)

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
    #Load config and CLI arguments
    config = configparser.ConfigParser()
    config.read("config.ini")
    
    parser = argparse.ArgumentParser(description="Analyze subreddit network aspects")
    parser.add_argument("--subreddit", type=str, help="Subreddit to analyze")
    args = parser.parse_args()
    
    subreddit_name = args.subreddit or config["defaults"].get("subreddit")
    if not subreddit_name:
        subreddit_name = input("Enter subreddit for network analysis: ").strip()
        if not subreddit_name:
            raise ValueError("Subreddit name is required.")
    
    subreddit_slug = subreddit_name.lower().replace(" ", "_")
    
    #Flairs (optional)
    default_flairs = config["defaults"].get("flairs", "").split(",")
    target_flairs = (
        [f.strip() for f in args.flairs.split(",")] if hasattr(args, "flairs") and args.flairs else
        [f.strip() for f in default_flairs if f.strip()]
    )    
    #Dynamic filenames
    posts_file = f"../data/{subreddit_slug}_posts.csv"
    comments_file = f"../data/{subreddit_slug}_comments.csv"
    filtered_posts_file = f"../data/{subreddit_slug}_sentiment_posts_filtered.csv"
    filtered_comments_file = f"../data/{subreddit_slug}_sentiment_comments_filtered.csv"
    
    #Check if filtered files exist BEFORE printing paths or trying to load
    if not os.path.exists(filtered_posts_file) or not os.path.exists(filtered_comments_file):
        print("\n Filtered sentiment files not found!")
        print("Run analize_sentiment.py and analize_comment_sentiment.py first to generate them.")
        raise SystemExit("Missing required sentiment files.")
    
    #Safe to print and load
    print(f" Loading posts: {posts_file}")
    print(f" Loading comments: {comments_file}")
    print(f" Loading filtered posts: {filtered_posts_file}")
    print(f" Loading filtered comments: {filtered_comments_file}")
    
    #Load and preprocess all datasets
    datasets = load_and_preprocess_data(posts_file, comments_file, filtered_posts_file, filtered_comments_file)
    posts_df = datasets['posts']
    comments_df = datasets['comments']
    filtered_posts = datasets['filtered_posts']
    filtered_comments = datasets['filtered_comments']
    
    #Graph building + analysis
    post_centric_graph = build_post_centric_graph(posts_df, comments_df)
    metrics = compute_graph_metrics(post_centric_graph)
    print_graph_analysis(metrics)
    
    #bi_graph definition and position of the nodes definition
    bi_graph_filtered_data = build_bipartite_graph(filtered_posts, filtered_comments)
    pos = visualize_bipartite_graph(bi_graph_filtered_data, "Refined Bipartite Graph Layout")
    
    #After building bipartite graph
    if bi_graph_filtered_data.number_of_nodes() < 5:
        print(f"\n⚠️ Graph too small for meaningful analysis ({bi_graph_filtered_data.number_of_nodes()} nodes).")
        print("✅ Skipping plots and metrics. Consider scraping a larger dataset.")
        return  # Exit early

    
    #Sentiment visualization
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
    
    #Post's engagement analysis
    engagement_stats = analyze_post_engagement(filtered_comments)
    print("High Engagement Post Summary:")
    print(engagement_stats.head())

    #Heatmap
    plot_comment_heatmap(filtered_comments)
    
    top_commenters_list = get_top_commenters(filtered_comments, top_k=20)
    commenter_network = build_commenter_network(filtered_comments, top_commenters_list)

    mini_cluster_graph = visualize_largest_cluster(commenter_network)
    
    #Mini cluster centrality 
    compute_mini_cluster_centrality(mini_cluster_graph)

    #Shared posts among the mini-cluster members
    cluster_nodes = set(mini_cluster_graph.nodes())
    identify_shared_posts(filtered_comments, filtered_posts, cluster_nodes)
    
    plot_sentiment_distribution(filtered_comments, cluster_nodes)
    
    #Commenter-commenter graph building
    commenter_graph = nx.Graph()
    for post_id, group in filtered_comments.groupby('post_id'):
        commenters = group['author'].unique()
        for i in range(len(commenters)):
            for j in range(i + 1, len(commenters)):
                u, v = commenters[i], commenters[j]
                if not commenter_graph.has_edge(u, v):
                    commenter_graph.add_edge(u, v, weight=0)
                commenter_graph[u][v]['weight'] += 1

    #Assign initial sentiments
    assign_initial_sentiments(filtered_comments, commenter_graph)

    #Sentiment propagation
    sentiment_evolution = propagate_sentiments(commenter_graph, max_steps=50)

    #Visualiza evolution of sentiments
    plot_sentiment_evolution(sentiment_evolution)
    
    #Weights shift for community detection as weights cannot be negative with the used algorithm
    shifted_graph = shift_edge_weights(commenter_graph)

    #Louvain community detection
    from community import community_louvain
    partition = community_louvain.best_partition(shifted_graph)

    #Assign 'community' attribute
    nx.set_node_attributes(commenter_graph, partition, name='community')

    #Analyze sentiments distribution per community
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

    #Highlight influencer nodes in subgraph
    visualize_communities(
        influencers["subgraph"],
        partition,  #Same partition as main graph
        highlight_nodes=top_nodes,
        title="Top Nodes Highlighted by Community"
    )
    
    #Compute modularity to check if the community detection algorithm worked properly
    modularity = compute_modularity(commenter_graph, partition)
    print(modularity)

    #Compute sentiment flow
    sentiment_flow_matrix = compute_sentiment_flow_matrix(commenter_graph, partition)
    print("Sentiment Flow Matrix:")
    print(sentiment_flow_matrix)

    #Heatmap visualization
    visualize_sentiment_flow(sentiment_flow_matrix)

    #Compute inter and intra community fluxes
    inter_flows, intra_flows = compute_flow_values(sentiment_flow_matrix)

    print("Top Inter-Community Flows:")
    for i, j, flow in inter_flows[:10]:
        print(f"Community {i} ↔ Community {j}: {flow:.2f}")

    print("Top Intra-Community Flows:")
    for i, flow in intra_flows[:10]:
        print(f"Community {i}: {flow:.2f}")

if __name__ == "__main__":
    main()
