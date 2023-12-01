import numpy as np
import pandas as pd

from typing import List, Dict


def calc_sample_rate(df: pd.DataFrame, time_column: str = 'Sec_Since_Start') -> float:
    """
    Calculates the sample rate based on the time intervals between consecutive measurements.

    Args:
        df (pd.DataFrame): A DataFrame containing the time values.
        time_column (str): The name of the column in the DataFrame that contains time values.

    Returns:
        float: The calculated sample rate in Hz.
    """
    time_values = df[time_column].values
    time_deltas = np.diff(time_values)
    mean_time_delta = np.mean(time_deltas)
    return 1 / mean_time_delta if mean_time_delta else float('inf')


def calc_amplitude(df: pd.DataFrame, column_name: str) -> float:
    """
    Calculates the amplitude of the specified column data in a DataFrame.

    The amplitude is defined as half the difference between the maximum and minimum values of the column data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column in the DataFrame for which to calculate the amplitude.

    Returns:
        float: The calculated amplitude.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame or if max and min values are not computable.

    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")

    try:
        max_value = df[column_name].max()
        min_value = df[column_name].min()
        amplitude = (max_value - min_value) / 2
        return amplitude
    except Exception as e:
        raise ValueError(f"Unable to calculate amplitude: {e}")


def calc_min_max(df: pd.DataFrame, sensor_column: str, time_column: str) -> dict:
    """
    Calculates the minimum and maximum values of a specified column in a DataFrame,
    along with their corresponding indices and times.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        sensor_column (str): The column name for which to find the min and max.
        time_column (str): The column name that contains the time values.

    Returns:
        dict: A dictionary with min and max information, each containing 'idx', 'time', and 'value'.
    """
    try:
        max_idx = df[sensor_column].idxmax()
        min_idx = df[sensor_column].idxmin()
        max_value = df[sensor_column].max()
        min_value = df[sensor_column].min()
        max_time = df[time_column].iloc[max_idx]
        min_time = df[time_column].iloc[min_idx]

        return {
            'max': {'idx': max_idx, 'time': max_time, 'value': max_value},
            'min': {'idx': min_idx, 'time': min_time, 'value': min_value}
        }
    except Exception as e:
        # You might want to handle specific exceptions or perform logging here
        raise RuntimeError(f"Error calculating min and max: {e}")


def convert_dict_to_list(param_dict: Dict[str, tuple]) -> List[tuple]:
    """
    Converts a dictionary of parameters to a list.

    Args:
    param_dict (Dict[str, float]): Dictionary of parameters.

    Returns:
    List[float]: List of parameter values.
    """
    return [param_dict[key] for key in param_dict]
