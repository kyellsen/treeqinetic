import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from ..analyse.fitting_functions import damped_osc


def extract_peak_valley_info(peaks: List[Dict[str, float]], valleys: List[Dict[str, float]]) -> (
        Tuple)[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def plot_osc_fit(data: pd.DataFrame, data_orig: pd.DataFrame, sensor_name: str, measurement_id: int, param_labels: List,
                 param_optimal: np.ndarray,
                 metrics: dict = None, metrics_warning: bool = False, peaks: List = None, valleys: List = None):
    # Berechnung der angepassten Schwingung

    time_correct = (data_orig['Sec_Since_Start'].max() - data['Sec_Since_Start'].max())
    data_orig['Sec_Since_Start'] = data_orig['Sec_Since_Start'] - time_correct

    time_for_fit = np.linspace(0, data['Sec_Since_Start'].max(), 2500)
    fitted_oscillation = damped_osc(time_for_fit, *param_optimal)

    fig = plt.figure()
    plt.plot(data['Sec_Since_Start'], data[sensor_name], color='black', zorder=2, label="Modifiziert Messwerte")
    # plt.scatter(data['Sec_Since_Start'], data[sensor_name], marker='.', s=5, c='blue', zorder=1, alpha=0.5,
    #             label="New Points")
    # plt.plot(data_orig['Sec_Since_Start'], data_orig[sensor_name], color='grey', zorder=1, label="Original Messwerte")
    plt.scatter(data_orig['Sec_Since_Start'], data_orig[sensor_name], marker='*', s=10, c='blue', zorder=4, alpha=1,
                label="Original Messpunkte")

    if peaks is not None and valleys is not None:
        peak_times, peak_values, valley_times, valley_values = extract_peak_valley_info(peaks, valleys)

        peak_times = peak_times - time_correct
        valley_times = valley_times - time_correct
        plt.scatter(peak_times, peak_values, c='red', marker='v', zorder=5, label='Peaks')
        plt.scatter(valley_times, valley_values, c='green', marker='^', zorder=5, label='Valleys')

    plt.plot(time_for_fit, fitted_oscillation, label="Angepasste Funktion", color="red", zorder=3)
    plt.title(f"Anpassung Schwingung an Messung ID {measurement_id} - {sensor_name}", fontsize=14)
    plt.xlabel("Zeit $t$ [s]")
    plt.ylabel(f"Absolute Randfaserdehnung $\\Delta$L [$\\mu$m] / Neigung $\\varphi$ [°]")

    # Parameterbezeichnungen und deren Werte formatieren
    param_optimal_dict = {label: param for label, param in zip(param_labels, param_optimal)}

    param_text = "Bestimmte Parameter:\n"
    # Ergänzt param_text um alle Parameter aus optimal_param_dict
    for label, value in param_optimal_dict.items():
        param_text += f"{label}: {value:.2f}\n"

    # Ergänzt param_text um alle Metriken aus dem metrics dict, falls vorhanden
    if metrics is not None:
        param_text += "\nQualitätsmetriken:\n"
        # Anzeige von "Warning" in rot und fett, falls metrics_warning True ist
        for metric, value in metrics.items():
            param_text += f"{metric}: {value:.2f}\n"
        if metrics_warning:
            param_text += f"\n! WARNUNG ! \nQualität ungenügend!\n\n"

    # Hinzufügen der formatierten Parameter zum Plot in der oberen rechten Ecke
    plt.annotate(param_text, xy=(0.95, 0.95), xycoords='axes fraction', fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black'),
                 verticalalignment='top', horizontalalignment='right')

    plt.legend(loc="lower right")
    return fig
