# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 18:00:53 2025

@author: Raffaele
"""

import networkx as nx
import pandas as pd

from network_aspects import assign_initial_sentiments  

def test_assign_initial_sentiments_classifies_mean_correctly():
    # GIVEN a graph with 3 nodes and comments with varying sentiment_body values
    graph = nx.Graph()
    graph.add_nodes_from(["user_pos", "user_neg", "user_neu"])

    data = {
        "author": ["user_pos", "user_pos", "user_neg", "user_neg", "user_neu"],
        "sentiment_body": [0.9, 0.8, -0.6, -0.7, 0.0]
    }
    df = pd.DataFrame(data)

    # WHEN assigning initial sentiments
    assigned = assign_initial_sentiments(df, graph)

    # THEN the correct sentiment values should be assigned
    assert assigned["user_pos"] == 1
    assert assigned["user_neg"] == -1
    assert assigned["user_neu"] == 0
    assert graph.nodes["user_pos"]["sentiment"] == 1
    assert graph.nodes["user_neg"]["sentiment"] == -1
    assert graph.nodes["user_neu"]["sentiment"] == 0
