import numpy as np
import pandas as pd
from typing import Tuple, List
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def damped_osc(time: np.ndarray, initial_amplitude: float, damping_coeff: float, angular_frequency: float,
               phase_angle: float, y_shift: float) -> np.ndarray:
    """
    Damped oscillation function.

    Args:
    time (np.ndarray): Array of time values.
    initial_amplitude (float): Initial amplitude of the oscillation.
    damping_coeff (float): Damping coefficient.
    angular_frequency (float): Angular frequency of the oscillation.
    phase_angle (float): Phase angle.
    y_shift (float): Vertical shift of the oscillation.

    Returns:
    np.ndarray: Calculated values of the damped oscillation function for each time value.
    """
    function = initial_amplitude * np.exp(-damping_coeff * time) * np.cos(
        2 * np.pi * angular_frequency * time + phase_angle) + y_shift
    return function


def fit_damped_osc(data: pd.DataFrame, sensor_name: str, initial_param: List[float],
                   param_bounds: Tuple[List[float], List[float]]) -> np.ndarray:
    """
    Fits a damped oscillation function to given data and returns the optimal parameters.

    Args:
    data (pd.DataFrame): DataFrame containing the data to fit.
    sensor_name (str): Column name of the sensor data to be fitted.
    initial_param (List[float]): List of initial parameter guesses for the fitting function.
    param_bounds (Tuple[List[float], List[float]]): Tuple containing two lists of lower and upper bounds for each parameter.

    Returns:
    np.ndarray: Array of optimal parameters.
    """
    param_optimal, _ = curve_fit(damped_osc, data['Sec_Since_Start'], data[sensor_name], p0=initial_param,
                                 bounds=param_bounds, maxfev=100000)

    return param_optimal


def calc_metrics(data: pd.DataFrame, sensor_name: str, param_optimal: np.ndarray) -> Tuple:
    """
    Calculates various metrics for the fitted data.

    Args:
    data (pd.DataFrame): DataFrame containing the original data.
    sensor_name (str): Column name of the sensor data.
    optimal_param (np.ndarray): Optimal parameters from the curve fitting.

    Returns:
    Tuple: Calculated metrics (MSE, MAE, RMSE, R²) for the fitted data.
    """
    fitted_values = damped_osc(data['Sec_Since_Start'], *param_optimal)
    mse = mean_squared_error(data[sensor_name], fitted_values)
    mae = mean_absolute_error(data[sensor_name], fitted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(data[sensor_name], fitted_values)

    return mse, mae, rmse, r2
