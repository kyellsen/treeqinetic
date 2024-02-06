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
    y_shift (float): Vertical get_shifted_trunk_data of the oscillation.

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


from scipy.optimize import minimize

from scipy.optimize import minimize

def mae_loss(params, time, sensor_data):
    """
    Berechnet den mittleren absoluten Fehler zwischen den Daten und dem Modell.

    Args:
        params (array): Modellparameter.
        time (np.ndarray): Zeitwerte.
        sensor_data (np.ndarray): Sensorwerte.

    Returns:
        float: MAE zwischen den Modellvorhersagen und den tatsächlichen Sensorwerten.
    """
    # Stellen Sie sicher, dass params als separate Argumente übergeben werden
    initial_amplitude, damping_coeff, angular_frequency, phase_angle, y_shift = params
    predicted = damped_osc(time, initial_amplitude, damping_coeff, angular_frequency, phase_angle, y_shift)
    mae = np.mean(np.abs(sensor_data - predicted))
    return mae

def fit_damped_osc_mae(data: pd.DataFrame, sensor_name: str, initial_param: List[float],
                       param_bounds: Tuple[List[float], List[float]]) -> np.ndarray:
    """
    Passt eine gedämpfte Schwingungsfunktion an die Daten an, wobei der MAE als Qualitätskriterium verwendet wird.

    Args:
        data (pd.DataFrame): DataFrame mit den zu fittenden Daten.
        sensor_name (str): Spaltenname der Sensordaten.
        initial_param (List[float]): Liste der Anfangsschätzungen für die Parameter.
        param_bounds (Tuple[List[float], List[float]]): Tuple mit den unteren und oberen Grenzen für jeden Parameter.

    Returns:
        np.ndarray: Optimierte Parameter.
    """
    # Konvertierung der Grenzen in das erforderliche Format für scipy.optimize.minimize
    bounds = [(low, high) for low, high in zip(*param_bounds)]

    result = minimize(mae_loss, np.array(initial_param), args=(data['Sec_Since_Start'], data[sensor_name]), bounds=bounds)
    return result.x
