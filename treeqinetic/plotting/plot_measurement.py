import matplotlib.pyplot as plt
import pandas as pd

from typing import List


def plot_multi_sensors(data: pd.DataFrame, file_name: str, measurement_id: int, sensor_names: list):
    fig, ax = plt.subplots()
    for sensor_name in sensor_names:
        ax.plot(data['Sec_Since_Start'], data[sensor_name], label=sensor_name)
        ax.scatter(data['Sec_Since_Start'], data[sensor_name], s=0.2, c='black', zorder=2)
    ax.set_xlabel('Time [Sec]')
    ax.set_ylabel('Elongation/Inclination [µm/°]')
    ax.legend()
    ax.set_title("Elongation/Inclination vs. Time", fontsize=16, ha='center')
    fig.suptitle(f"File '{file_name}' Measurement ID '{measurement_id}'", fontsize=12, ha='center')

    return fig


def configure_plot(ax, x_data: pd.Series, y_data: pd.Series, sensor_name: str, x_label: str, y_label: str, title: str,
                   color: str = 'b') -> None:
    """
    Configures a matplotlib plot for oscillation.

    Parameters:
    ax (matplotlib.axes.Axes): The axes object to configure.
    x_data (pd.Series): The data for the x-axis.
    y_data (pd.Series): The data for the y-axis.
    sensor_name (str): The name of the sensor.
    x_label (str): Label for the x-axis.
    y_label (str): Label for the y-axis.
    title (str): Title for the plot.
    color (str, optional): Color for the plot. Default is blue.
    """
    ax.plot(x_data, y_data, color=color)
    ax.scatter(x_data, y_data, s=5, c='black', zorder=2)
    ax.set_title(title.format(sensor_name))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def plot_oscillation(ax1, ax2, data, sensor_name, oscillation_data_orig):
    """
    Configures and adjusts two subplots for oscillation data visualization.

    Parameters:
    ax1, ax2 (matplotlib.axes.Axes): Axes objects for the plots.
    data (pd.DataFrame): DataFrame containing the main data.
    sensor_name (str): The name of the sensor.
    oscillation_data_orig (pd.DataFrame): DataFrame containing the original oscillation data.
    """
    configure_plot(ax1, data['Sec_Since_Start'], data[sensor_name], sensor_name, 'Time [Sec]', sensor_name,
                   'Original Data ({})')

    if oscillation_data_orig is not None:
        configure_plot(ax2, oscillation_data_orig['Sec_Since_Start'], oscillation_data_orig[sensor_name], sensor_name,
                       'Time [Sec]', sensor_name, 'Isolated Oscillation {}')

    # Equalizing the Y-axes
    min_y = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(min_y, max_y)
    ax2.set_ylim(min_y, max_y)


def plot_select_oscillation_single(data: pd.DataFrame, sensor_name: str,
                                   oscillation_data_orig: pd.DataFrame) -> plt.Figure:
    """
    Plots selected oscillation data for a single sensor.

    Parameters:
    data (pd.DataFrame): The main data DataFrame.
    sensor_name (str): The name of the sensor to plot.
    oscillation_data_orig (pd.DataFrame): The DataFrame of original oscillation data.

    Returns:
    plt.Figure: The matplotlib figure object for the plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    plot_oscillation(ax1, ax2, data, sensor_name, oscillation_data_orig)
    return fig


def plot_select_oscillation_multi(data: pd.DataFrame, sensor_names: List[str],
                                  oscillations_data_orig: List[pd.DataFrame]) -> plt.Figure:
    """
    Plots selected oscillation data for multiple sensors.

    Parameters:
    data (pd.DataFrame): The main data DataFrame.
    sensor_names (List[str]): List of sensor names to plot.
    oscillations_data_orig (List[pd.DataFrame]): List of DataFrames of original oscillation data for each sensor.

    Returns:
    plt.Figure: The matplotlib figure object for the plot.
    """
    num_sensors = len(sensor_names)
    fig, axs = plt.subplots(num_sensors, 2, figsize=(15, 6 * num_sensors))
    axs = axs if num_sensors > 1 else [axs]

    for i, sensor_name in enumerate(sensor_names):
        ax1, ax2 = axs[i]
        plot_oscillation(ax1, ax2, data, sensor_name, oscillations_data_orig[i])

    return fig
