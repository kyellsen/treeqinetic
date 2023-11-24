import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional

from ..analyse.fitting_functions import exp_decreasing, damped_osc


def extract_peak_valley_info(peaks: List[Dict[str, float]], valleys: List[Dict[str, float]]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts time and value information from peaks and valleys.

    Parameters:
    peaks (List[Dict[str, float]]): List of dictionaries with 'time' and 'value' keys for peaks.
    valleys (List[Dict[str, float]]): List of dictionaries with 'time' and 'value' keys for valleys.

    Returns:
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Arrays of times and values for peaks and valleys.
    """
    peak_times = np.array([peak['time'] for peak in peaks])
    peak_values = np.array([peak['value'] for peak in peaks])
    valley_times = np.array([valley['time'] for valley in valleys])
    valley_values = np.array([valley['value'] for valley in valleys])

    return peak_times, peak_values, valley_times, valley_values


def plot_oscillation(data: pd.DataFrame, sensor_name: str, with_peaks: bool, peaks: List[Dict[str, float]],
                     valleys: List[Dict[str, float]], max_value: float, min_value: float, with_amplitude: bool,
                     amplitude: Optional[float], amplitude_2: Optional[float], with_damping: bool,
                     damping_coeff_peaks: Optional[float], damping_coeff_valleys: Optional[float]) -> plt.Figure:
    """
    Plots oscillation data with optional features like peaks, valleys, amplitude, and damping.

    Parameters:
    data pd.DataFrame: Oscillation data.
    sensor_name (str): Name of the sensor.
    with_peaks (bool): Whether to plot peaks or not.
    peaks (List[Dict[str, float]]): Peak data.
    valleys (List[Dict[str, float]]): Valley data.
    max_value (float): Maximum value for amplitude.
    min_value (float): Minimum value for amplitude.
    with_amplitude (bool): Whether to plot amplitude or not.
    amplitude (Optional[float]): Amplitude value.
    amplitude_2 (Optional[float]): Second amplitude value.
    with_damping (bool): Whether to plot damping or not.
    damping_coeff_peaks (Optional[float]): Damping coefficient for peaks.
    damping_coeff_valleys (Optional[float]): Damping coefficient for valleys.

    Returns:
    plt.Figure: The plotted figure.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data['Sec_Since_Start'], data[sensor_name], label='Data', alpha=0.7)
    ax.scatter(data['Sec_Since_Start'], data[sensor_name], s=10, c='black', zorder=2, alpha=0.7)

    peak_times, peak_values, valley_times, valley_values = extract_peak_valley_info(peaks, valleys)

    if with_peaks:
        ax.scatter(peak_times, peak_values, c='red', marker='v', zorder=3, label='Peaks')
        ax.scatter(valley_times, valley_values, c='green', marker='^', zorder=3, label='Valleys')
        ax.scatter(peak_times[1], peak_values[1], c='orange', marker='v', zorder=5, label='Peak1')

    if with_amplitude:
        if amplitude is not None:
            ax.errorbar(peak_times[0], (max_value + min_value) / 2, yerr=amplitude, fmt='none', ecolor='purple',
                        capsize=10, label='Amplitude', zorder=4)
        if amplitude_2 is not None:
            ax.errorbar(peak_times[1], (peak_values[1] + min_value) / 2, yerr=amplitude_2, fmt='none', ecolor='green',
                        capsize=10, label='Amplitude_2', zorder=4)

    if with_damping:
        min_time = min(min(peak_times), min(valley_times))
        max_time = max(max(peak_times), max(valley_times))
        time_values = np.linspace(min_time, max_time, 500)

        if damping_coeff_peaks is not None:
            ax.plot(time_values, exp_decreasing(time_values - min_time, peak_values[0], damping_coeff_peaks),
                    label='Damping (Peaks)', linestyle='--', alpha=0.7)
        if damping_coeff_valleys is not None:
            ax.plot(time_values, exp_decreasing(time_values - min_time, valley_values[0], damping_coeff_valleys),
                    label='Damping (Valleys)', linestyle='--', alpha=0.7)

    plt.title(f"Oscillation {sensor_name}")
    plt.xlabel("Time (Sec)")
    plt.ylabel(f"Elongation/Inclination [µm/°] {sensor_name}")
    plt.legend()

    return fig


def plot_oscillation_and_fit(data: pd.DataFrame, data_orig: pd.DataFrame, sensor_name: str, optimal_params, mse: float,
                             peaks=None, valleys=None):
    # Berechnung der angepassten Schwingung

    time_correct = (data_orig['Sec_Since_Start'].max() - data['Sec_Since_Start'].max())
    data_orig['Sec_Since_Start'] = data_orig['Sec_Since_Start'] - time_correct

    time_for_fit = np.linspace(0, data['Sec_Since_Start'].max(), 3000)
    fitted_oscillation = damped_osc(time_for_fit, *optimal_params)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(data['Sec_Since_Start'], data[sensor_name], color='black', zorder=2, label="Modified Measurement")
    plt.plot(data_orig['Sec_Since_Start'], data_orig[sensor_name], color='grey', zorder=1, label="Original Measurement")
    plt.scatter(data_orig['Sec_Since_Start'], data_orig[sensor_name], marker='*', s=10, c='red', zorder=3, alpha=1,
                label="Original Points")

    if peaks is not None and valleys is not None:
        peak_times, peak_values, valley_times, valley_values = extract_peak_valley_info(peaks, valleys)

        peak_times = peak_times - time_correct
        valley_times = valley_times - time_correct
        plt.scatter(peak_times, peak_values, c='red', marker='v', zorder=4, label='Peaks')
        plt.scatter(valley_times, valley_values, c='green', marker='^', zorder=4, label='Valleys')

    plt.plot(time_for_fit, fitted_oscillation, label="Fitted Function", color="green")
    plt.title(f"Oscillation {sensor_name} and fitted Funktion")
    plt.xlabel("Time (Sec)")
    plt.ylabel(f"Elongation/Inclination [µm/°] {sensor_name}")


    # Parameterbezeichnungen und deren Werte formatieren
    param_labels = ['Initial Amplitude', 'Damping Coeff', 'Angular Frequency', 'Phase Angle']
    params_text = "Optimal Params:\n"
    for label, value in zip(param_labels, optimal_params):
        params_text += f"{label}: {value:.2f}\n"

    # MSE hinzufügen
    params_text += f"MSE: {mse:.2f}"

    # Hinzufügen der formatierten Parameter und des MSE zum Plot
    plt.annotate(params_text, xy=(0.1, 0.95), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
                 verticalalignment='top')

    plt.grid(True)
    plt.legend()
    return fig
