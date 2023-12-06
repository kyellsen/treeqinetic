import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats


def plot_error_histogram(errors: np.ndarray, bins: int = 50, title: str = "Histogram of Errors",
                         show_plot: bool = False, trim_percent: float = 0.0) -> plt.Figure:
    """
    Plots a histogram of the fitting errors using Seaborn and returns the matplotlib figure.
    Optionally trims the outer x percent of data from both ends for the plot.

    Args:
    errors (np.ndarray): Array of residuals/errors.
    bins (int): Number of bins in the histogram.
    title (str): Title of the histogram plot.
    show_plot (bool): If True, display the plot. Defaults to True.
    trim_percent (float): Percent of data to trim from each end for plotting. Defaults to 0.0.

    Returns:
    plt.Figure: The matplotlib figure object for the plot.
    """
    if trim_percent > 0.0:
        lower_bound = np.percentile(errors, trim_percent)
        upper_bound = np.percentile(errors, 100 - trim_percent)
        errors = errors[(errors >= lower_bound) & (errors <= upper_bound)]
        title += f" (Trimmed {trim_percent}% each end)"

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(errors, bins=bins, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.grid(True)

    if show_plot:
        plt.show()

    return fig


def plot_error_qq(errors: np.ndarray, title: str = "Q-Q Plot of Errors", show_plot: bool = False) -> plt.Figure:
    """
    Plots a Q-Q plot of the fitting errors to assess normality and returns the matplotlib figure.

    Args:
    errors (np.ndarray): Array of residuals/errors.
    title (str): Title of the Q-Q plot.
    show_plot (bool): If True, display the plot. Defaults to True.

    Returns:
    plt.Figure: The matplotlib figure object for the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    stats.probplot(errors, dist="norm", plot=ax)
    ax.set_title(title)
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Ordered Values')
    ax.grid(True)

    if show_plot:
        plt.show()

    return fig


def plot_error_violin(errors_dict: dict, title: str = "Violin Plot of Errors", show_plot: bool = False) -> plt.Figure:
    """
    Creates a violin plot of errors for each key in the errors_dict.

    Args:
        errors_dict (dict): A dictionary where keys are categories (e.g., sensor names) and values are arrays of errors.
        title (str): Title of the plot.
        show_plot:

    Returns:
        plt.Figure: The created matplotlib figure.
    """
    # Vorbereiten der Daten für den Violin-Plot
    data_for_plotting = []
    for category, errors in errors_dict.items():
        for error in errors:
            data_for_plotting.append({'Category': category, 'Error': error})

    df_for_plotting = pd.DataFrame(data_for_plotting)

    # Erstellen des Violin-Plots
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='Category', y='Error', data=df_for_plotting, ax=ax, palette=sns.color_palette("deep", len(errors_dict)), legend=False)
    ax.set_title(title)
    ax.set_xlabel('Category')
    ax.set_ylabel('Error')

    if show_plot:
        plt.show()

    return fig
