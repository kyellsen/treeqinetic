import matplotlib.pyplot as plt
import pandas as pd

from typing import List, Dict, Union


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


def plot_select_oscillations(data: pd.DataFrame,
                             sensor_names: Union[str, List[str]],
                             oscillations_data_orig: Union[pd.DataFrame, Dict[str, pd.DataFrame]]):
    """
    Configures and adjusts multiple subplots for oscillation data visualization.

    Args:
        data (pd.DataFrame): The main data frame.
        sensor_names (Union[str, List[str]]): A single sensor name or a list of sensor names.
        oscillations_data_orig (Union[pd.DataFrame, Dict[str, pd.DataFrame]]): Oscillation data, either as a single DataFrame or a dictionary of DataFrames.
    """
    if isinstance(sensor_names, str):
        sensor_names = [sensor_names]  # Ensure sensor_names is always a list
        oscillations_data_orig = {sensor_names[0]: oscillations_data_orig}  # Wrap in dict if necessary

    num_sensors = len(sensor_names)
    fig, axes = plt.subplots(num_sensors, 2, figsize=(15, 6 * num_sensors))

    for i, sensor_name in enumerate(sensor_names):
        if sensor_name in oscillations_data_orig:
            ax1 = axes[i, 0] if num_sensors > 1 else axes[0]
            ax2 = axes[i, 1] if num_sensors > 1 else axes[1]

            configure_plot(ax1, data['Sec_Since_Start'], data[sensor_name], sensor_name,
                           'Time [Sec]', f"Elongation/Inclination [µm/°] {sensor_name}",
                           f'Original Data {sensor_name}')

            configure_plot(ax2, oscillations_data_orig[sensor_name]['Sec_Since_Start'],
                           oscillations_data_orig[sensor_name][sensor_name], sensor_name,
                           'Time [Sec]', f"Elongation/Inclination [µm/°] {sensor_name}",
                           f'Isolated Oscillation {sensor_name}')

            # Equalizing the Y-axes for each sensor
            min_y = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
            max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
            ax1.set_ylim(min_y, max_y)
            ax2.set_ylim(min_y, max_y)

    plt.tight_layout()
    return fig
