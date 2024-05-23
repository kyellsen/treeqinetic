import numpy as np
import pandas as pd
from typing import Tuple, List
from scipy.optimize import curve_fit

from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

def damped_osc(time: np.ndarray, initial_amplitude: float, damping_coeff: float, angular_frequency: float,
               phase_angle: float, y_shift: float, x_shift: float) -> np.ndarray:
    """
    Damped oscillation function with optional horizontal shift.

    Args:
    time (np.ndarray): Array of time values.
    initial_amplitude (float): Initial amplitude of the oscillation.
    damping_coeff (float): Damping coefficient.
    angular_frequency (float): Angular frequency of the oscillation.
    phase_angle (float): Phase angle.
    y_shift (float): Vertical shift of the oscillation.
    x_shift (float): Horizontal shift of the oscillation.

    Returns:
    np.ndarray: Calculated values of the damped oscillation function for each time value.
    """
    function = initial_amplitude * np.exp(-damping_coeff * (time - x_shift)) * np.cos(
        2 * np.pi * angular_frequency * (time - x_shift) + phase_angle) + y_shift
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

    result = minimize(mae_loss, np.array(initial_param), args=(data['Sec_Since_Start'], data[sensor_name]),
                      bounds=bounds)
    return result.x


def mae_loss(params, time, sensor_data):
    """
    Berechnet den mittleren absoluten Fehler zwischen den Daten und dem Modell.

    Args:
        params (array): Modellparameter [initial_amplitude, damping_coeff, angular_frequency, phase_angle, y_shift, x_shift].
        time (np.ndarray): Zeitwerte.
        sensor_data (np.ndarray): Sensorwerte.

    Returns:
        float: MAE zwischen den Modellvorhersagen und den tatsächlichen Sensorwerten.
    """
    # Stellen Sie sicher, dass params als separate Argumente übergeben werden
    initial_amplitude, damping_coeff, angular_frequency, phase_angle, y_shift, x_shift = params
    # Modellvorhersagen berechnen
    predicted = damped_osc(time, initial_amplitude, damping_coeff, angular_frequency, phase_angle, y_shift, x_shift)
    # Verwendung von sklearn's mean_absolute_error zur Berechnung des MAE
    mae = mean_absolute_error(sensor_data, predicted)
    return mae

