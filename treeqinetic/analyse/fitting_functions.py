import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from scipy.optimize import minimize, curve_fit, OptimizeResult
from sklearn.metrics import mean_absolute_error, mean_squared_error


def damped_osc(time: np.ndarray, initial_amplitude: float, damping_coeff: float, frequency: float,
               phase_angle: float, y_shift: float, x_shift: float) -> np.ndarray:
    """
    Damped oscillation function with optional horizontal shift.

    Args:
    time (np.ndarray): Array of time values.
    initial_amplitude (float): Initial amplitude of the oscillation.
    damping_coeff (float): Damping coefficient.
    frequency (float): Angular frequency of the oscillation.
    phase_angle (float): Phase angle.
    y_shift (float): Vertical shift of the oscillation.
    x_shift (float): Horizontal shift of the oscillation.

    Returns:
    np.ndarray: Calculated values of the damped oscillation function for each time value.
    """
    function = initial_amplitude * np.exp(-damping_coeff * (time - x_shift)) * np.cos(
        2 * np.pi * frequency * (time - x_shift) + phase_angle) + y_shift
    return function


def mae_loss(params, time, sensor_data):
    initial_amplitude, damping_coeff, frequency, phase_angle, y_shift, x_shift = params
    predicted = damped_osc(time, initial_amplitude, damping_coeff, frequency, phase_angle, y_shift, x_shift)
    mae = mean_absolute_error(sensor_data, predicted)
    return mae


def mse_loss(params, time, sensor_data):
    initial_amplitude, damping_coeff, frequency, phase_angle, y_shift, x_shift = params
    predicted = damped_osc(time, initial_amplitude, damping_coeff, frequency, phase_angle, y_shift, x_shift)
    mse = mean_squared_error(sensor_data, predicted)
    return mse


def fit_damped_osc(
    data: pd.DataFrame,
    sensor_name: str,
    initial_param: List[float],
    param_bounds: Tuple[List[float], List[float]],
    optimize_criterion: str = 'mae',
    options: Optional[Dict[str, Any]] = None
) -> OptimizeResult:
    """
    Fits a damped oscillation function to given data using either MAE or MSE as the optimization criterion.

    Args:
        data (pd.DataFrame): DataFrame containing the data to fit.
        sensor_name (str): Column name of the sensor data to be fitted.
        initial_param (List[float]): List of initial parameter guesses for the fitting function.
        param_bounds (Tuple[List[float], List[float]]): Tuple containing two lists of lower and upper bounds for each parameter.
        optimize_criterion (str): Criterion to optimize ('mae' or 'mse').
        options (dict, optional): Options dictionary for scipy.optimize.minimize. If None, default values are used.

    Returns:
        OptimizeResult: The full optimization result object from scipy.optimize.minimize,
                        which includes the optimized parameters in result.x as well as
                        additional information such as the number of iterations (result.nit),
                        the optimization status, and more.
    """
    # Falls keine eigenen Options übergeben wurden, auf Defaultwerte zurückgreifen
    if options is None:
        options = {
            'maxiter': 100000,
            'ftol': 2.220446049250313e-09,
        }

    bounds = [(low, high) for low, high in zip(*param_bounds)]

    if optimize_criterion == 'mae':
        loss_function = mae_loss
    elif optimize_criterion == 'mse':
        loss_function = mse_loss
    else:
        raise ValueError("Criterion must be 'mae' or 'mse'")

    result = minimize(
        loss_function,
        x0=np.array(initial_param),
        args=(data['Sec_Since_Start'], data[sensor_name]),
        bounds=bounds,
        method='L-BFGS-B',
        options=options
    )

    return result

