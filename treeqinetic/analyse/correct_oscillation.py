import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator


def zero_base_column(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Shifts the values in a specified column of a DataFrame such that the minimum value becomes zero.

    Args:
    df (pd.DataFrame): The DataFrame to be processed.
    column_name (str): The name of the column to be zero-based.

    Returns:
    pd.DataFrame: The DataFrame with the adjusted column.
    """

    if column_name in df.columns:
        min_value = df[column_name].min()
        df[column_name] = df[column_name] - min_value
        return df
    else:
        raise ValueError(f"Column '{column_name}' not found in DataFrame.")


def remove_values_above_percentage(df, column_name, amplitude_2, percentage):
    # Schwellenwert berechnen
    threshold = amplitude_2 * (percentage / 100) * 2

    # Entfernen der Werte, die größer als der Schwellenwert sind
    df.drop(df[df[column_name] > threshold].index, inplace=True)

    # Normalize the 'Sec_Since_Start' column by subtracting its minimum value
    df['Sec_Since_Start'] = df['Sec_Since_Start'] - df['Sec_Since_Start'].min()

    # Index des DataFrames neu ordnen
    df.reset_index(drop=True, inplace=True)

    return df


def clean_peaks_and_valleys(df, column_name, peaks, valleys):
    # Kombinieren und Sortieren der Peaks und Valleys
    combined = sorted(peaks + valleys, key=lambda x: x['index'])

    for i in range(len(combined) - 1):
        start_idx = combined[i]['index']
        end_idx = combined[i + 1]['index']
        start_value = df.iloc[start_idx][column_name]
        end_value = df.iloc[end_idx][column_name]

        # Bestimmen der Richtung: Anstieg oder Abfall
        is_increasing = end_value > start_value

        # Überprüfen und Entfernen der nicht kontinuierlichen Werte
        last_valid_value = start_value
        for idx in range(start_idx + 1, end_idx):
            current_value = df.iloc[idx][column_name]
            if (is_increasing and current_value < last_valid_value) or (
                    not is_increasing and current_value > last_valid_value):
                df.drop(idx, inplace=True)
            else:
                last_valid_value = current_value

    # Index des neuen DataFrames neu ordnen
    df.reset_index(drop=True, inplace=True)

    return df


def interpolate_points(df, column_name, sample_rate):
    # Erstellung des PchipInterpolators
    interpolator = PchipInterpolator(df['Sec_Since_Start'], df[column_name])

    # Berechnung der Gesamtzeit und der Anzahl der erforderlichen Punkte
    min_time = df['Sec_Since_Start'].min()
    max_time = df['Sec_Since_Start'].max()
    total_time = max_time - min_time
    num_points = int(total_time * sample_rate)

    # Erzeugung neuer Zeitpunkte für die Interpolation
    new_times = np.linspace(min_time, max_time, num=num_points)

    # Durchführung der Interpolation
    new_values = interpolator(new_times)

    # Erstellen des neuen DataFrames
    new_df = pd.DataFrame({'Sec_Since_Start': new_times, column_name: new_values})

    return new_df
