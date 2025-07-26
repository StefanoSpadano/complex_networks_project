# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:57:19 2025

@author: Raffaele
"""

import os
from datetime import datetime
import matplotlib.pyplot as plt
import inspect
import numpy as np
import pandas as pd


def save_plot(name_hint: str = "", folder: str = ""):
    """
    Saves the current matplotlib plot with an automatic filename based on timestamp and script name.

    Parameters:
    - name_hint (str): Optional description of the plot (e.g., 'degree_distribution')
    - folder (str): Folder to save plots into.
    """
    # Create folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Get script name (caller)
    frame = inspect.stack()[1]
    script_name = os.path.splitext(os.path.basename(frame.filename))[0]

    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compose file name
    name_parts = [script_name]
    if name_hint:
        name_parts.append(name_hint)
    name_parts.append(timestamp)
    file_name = "_".join(name_parts) + ".png"

    # Full path
    full_path = os.path.join(folder, file_name)

    # Save
    plt.savefig(full_path, bbox_inches="tight")
    print(f"✅ Plot saved as {full_path}")
    

def categorize_sentiment(score, threshold=0.1, epsilon=1e-18):
    #1e-18 stands for 1^(10)-18
    if score > threshold + epsilon:
        return 'Positive'
    elif score < -threshold - epsilon:
        return 'Negative'
    else:
        return 'Neutral'



def compute_flow_values(matrix):
    """
    Compute inter-community and intra-community flow values.

    Args:
        matrix (np.ndarray): The sentiment flow matrix.

    Returns:
        tuple: A tuple containing two lists:
            - inter_flows: List of (i, j, flow) for inter-community flows.
            - intra_flows: List of (i, flow) for intra-community flows.

    Raises:
        ValueError: If the matrix is empty or not square.
    """
    if matrix.size == 0:
        raise ValueError("Input matrix is empty.")
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square (NxN).")

    num_communities = matrix.shape[0]
    inter_flows = []
    intra_flows = []

    for i in range(num_communities):
        for j in range(num_communities):
            if i == j:
                intra_flows.append((i, matrix[i, j]))
            else:
                if matrix[i, j] > 0:
                    inter_flows.append((i, j, matrix[i, j]))

    inter_flows = sorted(inter_flows, key=lambda x: x[2], reverse=True)
    intra_flows = sorted(intra_flows, key=lambda x: x[1], reverse=True)

    return inter_flows, intra_flows



def load_data(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, lineterminator='\n', engine='python')


