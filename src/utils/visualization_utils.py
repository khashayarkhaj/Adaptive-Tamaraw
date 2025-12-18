# a set of functions used to visualize different elements like traces

import numpy as np
import matplotlib.pyplot as plt
from utils.file_operations import save_file
import os

def visualize_tam(tams, website_index='', super_matrix=False, save_dir=None, trace_num=0, titles=None, 
                 regulator_configs=None, overheads=None, surges_nums=None, show_title = False):
    """
    Given TAMs (traffic aggregation matrices), visualize them with similar palette.

    Parameters
    ----------
        tams (list): 1 to 4 traces which are at the tam level (shape [2,n]).
        website_index (index or list): the class indexes that the traces belong to
        super_matrix: whether this trace is the super matrix of its class
        save_dir: if given, the figure will be saved at save_dir/website_index.png
        trace_num (index or list): if not supermatrix, report trace index from class
        titles (str or list): Optional custom titles for the plots. For single plot,
            pass a string. For multiple plots, pass a list of strings.
        regulator_configs (list of dict or None): Optional list of regulator configuration
            dictionaries containing parameters like ORIG_RATE, DEPRECIATION_RATE, etc.
        overheads (list of dict or None): Optional list of overhead dictionaries containing
            bandwidth and time overhead metrics
        surges_nums (list of int or None): Optional list of surge numbers
    """
    num_tams = len(tams)
    if num_tams > 4:
        raise ValueError("Maximum 4 TAMs can be visualized at once")
    
    # Handle website indices
    if website_index != '':
        if isinstance(website_index, (list, tuple)):
            website_indices = website_index[:num_tams]
        else:
            website_indices = [website_index] * num_tams
    else:
        website_indices = [''] * num_tams
        
    # Handle trace numbers
    if not isinstance(trace_num, (list, tuple)):
        trace_nums = [trace_num] * num_tams
    else:
        trace_nums = trace_num[:num_tams]
        
    # Handle regulator configs
    if regulator_configs is None:
        regulator_configs = [None] * num_tams
    elif isinstance(regulator_configs, dict):
        regulator_configs = [regulator_configs] * num_tams
        
    # Handle overheads
    if overheads is None:
        overheads = [None] * num_tams
    elif isinstance(overheads, dict):
        overheads = [overheads] * num_tams
        
    # Handle surges
    if surges_nums is None:
        surges_nums = [None] * num_tams
    elif isinstance(surges_nums, (int, float)):
        surges_nums = [surges_nums] * num_tams
    
    # Set up the subplot grid
    if num_tams == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes = [axes]
    elif num_tams == 2:
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
    elif num_tams == 3:
        fig, axes = plt.subplots(1, 3, figsize=(30, 5))
    else:  # num_tams == 4
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        axes = axes.flatten()

    # Plot each TAM
    for i, (tam, ax) in enumerate(zip(tams, axes)):
        # Extract outgoing and incoming packets
        outgoing_packets = tam[0]
        incoming_packets = -tam[1]

        # Create the plot
        ax.fill_between(range(tam.shape[1]), outgoing_packets, 
                       step="pre", label="Outgoing Packets", color="blue")
        ax.fill_between(range(tam.shape[1]), incoming_packets, 
                       step="pre", label="Incoming Packets", color="red")
        
        # Prepare text for configs, overheads, and surges
        text_blocks = []
        
        # Add regulator config parameters
        if regulator_configs[i]:
            config = regulator_configs[i]
            config_text = '\n'.join([
                f"R={config.get('ORIG_RATE', 'N/A'):.2f}",
                f"D={config.get('DEPRECIATION_RATE', 'N/A'):.2f}",
                f"T={config.get('BURST_THRESHOLD', 'N/A'):.2f}",
                f"U={config.get('UPLOAD_RATIO', 'N/A'):.2f}",
                f"C={config.get('DELAY_CAP', 'N/A'):.2f}",
                f"N'={(config.get('budget_used', 'N/A'))}"
            ])
            text_blocks.append(config_text)
            
        # Add surge number if provided
        if surges_nums[i] is not None:
            surge_text = f"surge={int(surges_nums[i])}"
            text_blocks.append(surge_text)
            
        # Add overhead metrics if provided
        if overheads[i]:
            oh = overheads[i]
            oh_text = '\n'.join([
                f"bw oh={oh.get('bw oh', 'N/A'):.2f}",
                f"bw oh out={oh.get('bw oh out', 'N/A'):.2f}",
                f"bw oh in={oh.get('bw oh in', 'N/A'):.2f}",
                f"time oh={oh.get('time oh', 'N/A'):.2f}"
            ])
            text_blocks.append(oh_text)
        
        # Combine all text blocks with spacing
        if text_blocks:
            combined_text = '\n\n'.join(text_blocks)
            
            # Place text box in top right inside plot (increased font size)
            ax.text(0.95, 0.95, combined_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3),
                   fontsize=20)  # Increased from 8 to 12
        
        # Set labels with larger font sizes
        ax.set_xlabel("Time Slots", fontsize=30)
        ax.set_ylabel("Pkt. Number", fontsize=30)
        
        # Set tick label font sizes
        ax.tick_params(axis='both', which='major', labelsize=28)
        
        # Set title with larger font size
        if show_title:
            if titles is not None:
                if isinstance(titles, (list, tuple)):
                    ax.set_title(titles[i], fontsize=16)
                else:
                    ax.set_title(titles if i == 0 else '', fontsize=16)
            else:
                if super_matrix:
                    ax.set_title(f"Super-Matrix of website {website_indices[i]}", fontsize=16)
                else:
                    ax.set_title(f"Random {trace_nums[i]}th trace of website {website_indices[i]}", fontsize=16)
        
        ax.grid(True)
        # Place legend at bottom right inside plot with larger font
        ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.02),
                 bbox_transform=ax.transAxes,
                 framealpha=0.8, fontsize=26)  # Added fontsize=12

    plt.tight_layout()
    
    # Save or show the plot
    if save_dir:
        if super_matrix:
            indices_str = 'vs'.join(str(idx) for idx in website_indices)
        else:
            indices_str = 'vs'.join(f'{idx}_{num}' for idx, num in zip(website_indices, trace_nums))
        file_name = f'{indices_str}.png'
        save_file(file_name=file_name, dir_path=save_dir)
    else:
        plt.show()