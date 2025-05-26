# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:57:19 2025

@author: Raffaele
"""

import os
from datetime import datetime
import matplotlib.pyplot as plt
import inspect

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
    

def categorize_sentiment(score):
    if score < -0.1:
        return 'Negative'
    elif score > 0.1:
        return 'Positive'
    else:
        return 'Neutral'

