import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


def exp_decreasing(time, initial_amplitude, damping_coeff):
    return initial_amplitude * np.exp(-damping_coeff * time)


def damped_osc(time, initial_amplitude, damping_coeff, angular_frequency, phase_angle):
    return initial_amplitude * np.exp(-damping_coeff * time) * np.cos(
        2 * np.pi * angular_frequency * time + phase_angle)


# Angepasste Funktion zur Anpassung der Schwingungsfunktion an die Messdaten
def fit_damped_osc(data, sensor_name, initial_params, bounds):
    # Nicht-lineare Kurvenanpassung mit Grenzwerten
    optimal_params, _ = curve_fit(damped_osc, data['Sec_Since_Start'], data[sensor_name], p0=initial_params,
                                  bounds=bounds, maxfev=1000000)
    return optimal_params


# Funktion zur Berechnung des MSE
def calc_mse(data, sensor_name, optimal_params):
    fitted_values = damped_osc(data['Sec_Since_Start'], *optimal_params)
    mse = mean_squared_error(data[sensor_name], fitted_values)
    return mse


# Hilfsfunktion für die umwandlung der dicts in list
def convert_dict_to_list(param_dict):
    return [param_dict[key] for key in param_dict]
