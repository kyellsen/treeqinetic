import numpy as np

from kj_logger import get_logger
logger = get_logger(__name__)


def calculate_slope(df, column_name):
    """

    Calculate the slope for a specific column in the DataFrame.

    """
    logger.debug(f"Calculating slope for column {column_name}.")
    try:
        slope = np.diff(df[column_name]) / np.diff(df['Sec_Since_Start'])
        df[f'Slope_{column_name}'] = np.append([np.nan], slope)
        logger.debug("Slope calculated successfully.")
    except Exception as e:
        logger.critical(f"Could not calculate slope for column {column_name}. Error: {e}")


# Modify the find_oscillation_start function to work directly with the original index
def find_oscillation_start(df, column_name, threshold_slope, min_time, min_value):
    """
    Find the start index of the oscillation period for a specific column.
    """
    logger.debug(f"Finding the start index of the oscillation period for column {column_name}.")

    # Filter the DataFrame based on the minimum time, without resetting the index
    df_filtered = df[df['Sec_Since_Start'] > min_time]
    start_index = None

    # Find the first point where the value falls below zero after min_time
    below_zero_index = df_filtered.index[df_filtered[column_name] <= 0].tolist()

    if not below_zero_index:
        logger.warning(f"Could not find any point where the value is below zero for column {column_name}.")
        return None

    first_below_zero_index = below_zero_index[0]

    # Loop backwards from the first_below_zero_index to find where the slope first falls below the threshold
    for i in range(first_below_zero_index, df_filtered.index.min(), -1):
        if df_filtered.loc[i, f'Slope_{column_name}'] > threshold_slope:
            if df_filtered.loc[i, column_name] > min_value:
                start_index = i
                break

    if start_index is not None:
        logger.debug(f"Start index of oscillation found at index {start_index} for column {column_name}.")
    else:
        logger.warning(f"Could not find a suitable start point for the oscillation period for column {column_name}.")

    return start_index


def extract_oscillation(df, column_name, start_index, duration):
    """
    Extract the oscillation period for a specific column including the entire steep drop.
    """
    logger.debug(f"Extracting the oscillation period for column {column_name}.")
    try:
        end_index = df.index[df['Sec_Since_Start'] >= df.at[start_index, 'Sec_Since_Start'] + duration].min()
        oscillation_df = df.loc[start_index:end_index, ['Sec_Since_Start', column_name]]
        logger.debug("Oscillation period extracted successfully.")
        return oscillation_df

    except Exception as e:
        logger.error(f"Could not extract oscillation period for column {column_name}. Error: {e}")
        return None
