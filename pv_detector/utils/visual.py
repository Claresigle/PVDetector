# -*-coding:utf-8-*-
import os

import matplotlib.pyplot as plt
import numpy as np


def visualize_array_as_bar(
        array,
        y_label="Value",
        title="Array Visualization",
        y_min=None,
        y_max=None,
        save_dir=None,
        figsize=(12, 8),
        bar_color="#050374",
        edge_color="white",
        grid=True,
        x_tick_interval=1):
    """Visualize a one-dimensional array as a layer-indexed bar chart."""
    array = np.asarray(array)
    if array.ndim != 1:
        raise ValueError("Input array must be 1-dimensional.")

    x_indices = np.arange(len(array))
    plt.rcParams.update({
        "font.family": "serif",
        "axes.unicode_minus": False,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    _, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(
        x_indices,
        array,
        color=bar_color,
        edgecolor=edge_color,
        linewidth=0.8,
    )
    ax.set_xticks(x_indices[::x_tick_interval])
    ax.set_xticklabels(x_indices[::x_tick_interval], rotation=0)

    for bar, value in zip(bars, array):
        height = bar.get_height()
        bar_width = bar.get_width()
        if height > max(array) * 0.05:
            ax.text(
                bar.get_x() + bar_width / 2,
                height / 2,
                f"{value:.2f}",
                ha="center",
                va="center",
                rotation=90,
                fontsize=9,
                color="white",
                fontweight="bold",
                linespacing=1.2,
            )

    ax.set_xlabel("Layer Index", fontweight="bold")
    ax.set_ylabel(y_label, fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=20)

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)
    else:
        ax.set_ylim(bottom=0, top=max(array) * 1.1)

    if grid:
        ax.grid(True, axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{title}.png")
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
            facecolor=ax.get_facecolor(),
        )
        print(f"Saved to: {save_path}")

    plt.close()
