import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def exp_decreasing(time: np.ndarray, initial_amplitude: float, damping_coeff: float) -> np.ndarray:
    """
    Exponential decreasing function.

    Args:
    time (np.ndarray): Array of time values.
    initial_amplitude (float): Initial amplitude of the exponential function.
    damping_coeff (float): Damping coefficient.

    Returns:
    np.ndarray: Calculated values of the exponential decreasing function for each time value.
    """
    return initial_amplitude * np.exp(-damping_coeff * time)


def damped_osc(time: np.ndarray, initial_amplitude: float, damping_coeff: float, angular_frequency: float, phase_angle: float, y_shift: float) -> np.ndarray:
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


def fit_damped_osc(data: pd.DataFrame, sensor_name: str, initial_params: List[float], param_bounds: Tuple[List[float], List[float]]) -> np.ndarray:
    """
    Fits a damped oscillation function to given data and returns the optimal parameters.

    Args:
    data (pd.DataFrame): DataFrame containing the data to fit.
    sensor_name (str): Column name of the sensor data to be fitted.
    initial_params (List[float]): List of initial parameter guesses for the fitting function.
    param_bounds (Tuple[List[float], List[float]]): Tuple containing two lists of lower and upper bounds for each parameter.

    Returns:
    np.ndarray: Array of optimal parameters.
    """
    params_optimal, _ = curve_fit(damped_osc, data['Sec_Since_Start'], data[sensor_name], p0=initial_params,
                                  bounds=param_bounds, maxfev=1000000)

    return params_optimal


def calc_metrics(data: pd.DataFrame, sensor_name: str, params_optimal: np.ndarray) -> Dict[str, float]:
    """
    Calculates various metrics for the fitted data.

    Args:
    data (pd.DataFrame): DataFrame containing the original data.
    sensor_name (str): Column name of the sensor data.
    optimal_params (np.ndarray): Optimal parameters from the curve fitting.

    Returns:
    Dict[str, float]: Calculated metrics (MSE, MAE, RMSE, R²) for the fitted data.
    """
    fitted_values = damped_osc(data['Sec_Since_Start'], *params_optimal)
    mse = mean_squared_error(data[sensor_name], fitted_values)
    mae = mean_absolute_error(data[sensor_name], fitted_values)
    rmse = np.sqrt(mse)
    r2 = r2_score(data[sensor_name], fitted_values)

    return {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}


def convert_dict_to_list(param_dict: Dict[str, float]) -> List[float]:
    """
    Converts a dictionary of parameters to a list.

    Args:
    param_dict (Dict[str, float]): Dictionary of parameters.

    Returns:
    List[float]: List of parameter values.
    """
    return [param_dict[key] for key in param_dict]
