import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from .base_class import BaseClass

from ..plotting import plot_oscillation
from ..analyse.fitting_functions import exp_decreasing
from ..analyse.correct_oscillation import zero_base_column, remove_values_above_percentage, clean_peaks_and_valleys, \
    interpolate_points
from ..analyse.fitting_functions import fit_damped_osc, calc_metrics, convert_dict_to_list

from kj_core import get_logger

logger = get_logger(__name__)


class Oscillation(BaseClass):
    def __init__(self, measurement, sensor_name, start_index, df: pd.DataFrame):
        super().__init__()

        self.measurement = measurement
        self.sensor_name = sensor_name  # = Column Name
        self.start_index = start_index
        self.df_orig = df

        # Reset the index of the DataFrame and make sure we don't lose the original index
        self.df = df.reset_index(drop=True, inplace=False)  # inplace=False -> creates a copy of df
        # Normalize the 'Sec_Since_Start' column by subtracting its minimum value
        self.df = zero_base_column(self.df, "Sec_Since_Start")

        self.df_fit = self.df.copy()

        # Initialize attributes with placeholder values
        self.offset = None
        self.sample_rate = None
        self.max_value = None
        self.min_value = None
        self.max_idx = None
        self.min_idx = None


        self.peaks = []
        self.valleys = []

        # params for manuell fitting
        self.amplitude = None
        self.amplitude_2 = None
        self.frequency = None
        self.damping_coeff_peaks = None
        self.damping_coeff_valleys = None
        self.damping_coeff_avg = None
        self.fit_data = None

        # params from automatic fitting
        self.initial_amplitude = None
        self.damping_coeff = None
        self.angular_frequency = None
        self.phase_angle = None
        self.y_shift = None

        # Perform calculations
        # self.calculate_offset()

        self.calculate_sample_rate()

        self.calculate_min_max()
        self.calculate_peaks_and_valleys()
        self.calculate_amplitude()
        self.calculate_amplitude_2()
        self.calculate_frequency()

        self.calculate_damping_coefficient()
        # self.fit_damped_oscillation()
        # self.plot_oscillation_fitting()

    def __str__(self):
        return f"Oscillation: '{self.measurement.file_name}', ID: {self.measurement.id}, Sensor: {self.sensor_name}'"

    def calculate_offset(self, window_size=5, last_percent=60):
        last_points = int(last_percent/100 * len(self.df))
        last_data = self.df[self.sensor_name][last_points:]
        offset = last_data.rolling(window=window_size).mean().dropna().mean()
        self.offset = offset
        self.df[self.sensor_name] -= self.offset

    # Methode zur Berechnung der Sample-Rate in Hz
    def calculate_sample_rate(self):
        # Zeitwerte aus dem DataFrame extrahieren
        time_values = self.df['Sec_Since_Start'].values
        # Zeitdifferenz zwischen aufeinanderfolgenden Messungen berechnen
        time_deltas = np.diff(time_values)
        # Durchschnittliche Zeitdifferenz berechnen
        mean_time_delta = np.mean(time_deltas)
        # Sample-Rate in Hz berechnen (1 / Durchschnittliche Zeitdifferenz)
        sample_rate = 1 / mean_time_delta
        # Speichern der Sample-Rate im Objekt
        self.sample_rate = sample_rate

    def calculate_min_max(self):
        try:
            self.max_value = self.df[self.sensor_name].max()
            self.min_value = self.df[self.sensor_name].min()
            self.max_idx = self.df[self.sensor_name].max()
            self.min_idx = self.df[self.sensor_name].min()
            logger.debug(f"Calculated min, max, idxmin, and idxmax for sensor {self.sensor_name}.")
        except Exception as e:
            logger.error(f"Failed to calculate min, max, idxmin, and idxmax for sensor {self.sensor_name}: {e}")

    def calculate_amplitude(self):
        try:
            self.amplitude = (self.max_value - self.min_value) / 2
            logger.debug(f"Calculated amplitude for sensor {self.sensor_name}.")
        except Exception as e:
            logger.error(f"Failed to calculate amplitude for sensor {self.sensor_name}: {e}")

    def calculate_peaks_and_valleys(self, min_distance_sec=1, search_range=0.75, prominence=5):
        min_distance = float(
            min_distance_sec * self.sample_rate)  # Mindestabstand von 1 Sekunde, umgerechnet in Samples
        cut_idx = int(len(self.df) * search_range)  # Index, der die ersten 75% der Daten kennzeichnet

        idxpeaks, _ = find_peaks(self.df[self.sensor_name][:cut_idx], distance=min_distance, prominence=prominence)
        idxvalleys, _ = find_peaks(-self.df[self.sensor_name][:cut_idx], distance=min_distance, prominence=prominence)

        for idx in idxpeaks:
            time = self.df['Sec_Since_Start'].iloc[idx]
            value = self.df[self.sensor_name].iloc[idx]
            self.peaks.append({'index': idx, 'time': time, 'value': value})

        for idx in idxvalleys:
            time = self.df['Sec_Since_Start'].iloc[idx]
            value = self.df[self.sensor_name].iloc[idx]
            self.valleys.append({'index': idx, 'time': time, 'value': value})

        # Überprüfen, ob der Wert des ersten Peaks kleiner als der des zweiten ist
        if len(self.peaks) > 1 and self.peaks[0]['value'] < self.peaks[1]['value']:
            # Ersten Peak entfernen
            del self.peaks[0]

        # Peak am Index 0 mit dem Wert self.max hinzufügen
        self.peaks.insert(0, {'index': 0, 'time': self.df['Sec_Since_Start'].iloc[0], 'value': self.max_value})

    def calculate_amplitude_2(self):
        try:
            self.amplitude_2 = (self.peaks[1]['value'] - self.min_value) / 2
            logger.debug(f"Calculated amplitude_2 for sensor {self.sensor_name}.")
        except Exception as e:
            self.amplitude_2 = self.amplitude / 2
            logger.warning(
                f"Failed to calculate amplitude_2 for sensor {self.sensor_name}. Using half of amplitude as fallback. Error: {e}")

    def calculate_frequency(self):
        # Extrahieren der Zeitwerte für Peaks und Valleys aus den Dictionaries
        peak_times = np.array([peak['time'] for peak in self.peaks])
        valley_times = np.array([valley['time'] for valley in self.valleys])

        time_deltas_peaks = np.diff(peak_times)
        time_deltas_valleys = np.diff(valley_times)
        mean_time_delta = np.mean(np.concatenate((time_deltas_peaks, time_deltas_valleys)))
        self.frequency = 1 / mean_time_delta

    def calculate_damping_coefficient(self):
        try:
            # Extract peak and valley times and values from the dictionaries
            peak_times = np.array([peak['time'] for peak in self.peaks])
            peak_values = np.array([peak['value'] for peak in self.peaks])
            valley_times = np.array([valley['time'] for valley in self.valleys])
            valley_values = np.array([valley['value'] for valley in self.valleys])

            # Fit exponential decaying function to peaks
            params_peak, _ = curve_fit(exp_decreasing, peak_times - peak_times[0], peak_values,
                                       p0=[peak_values[0], 0.1])
            initial_amplitude_peak, damping_coeff_peak = params_peak

            # Fit exponential decaying function to valleys
            params_valley, _ = curve_fit(exp_decreasing, valley_times - valley_times[0], valley_values,
                                         p0=[valley_values[0], 0.3])
            initial_amplitude_valley, damping_coeff_valley = params_valley

            # Calculate the average damping coefficient
            damping_coeff_avg = (damping_coeff_peak + damping_coeff_valley) / 2

            # Save the individual and average damping coefficients to the object
            self.damping_coeff_peaks = damping_coeff_peak
            self.damping_coeff_valleys = damping_coeff_valley
            self.damping_coeff_avg = damping_coeff_avg

        except Exception as e:
            logger.error(f"Failed to calculate damping coefficients: {e}")
            return None, None, None

    def plot_oscillation(self, with_peaks=True, with_amplitude=True, with_damping=True):
        """

        """
        try:
            fig = plot_oscillation.plot_oscillation(self.df, self.sensor_name, with_peaks, self.peaks, self.valleys,
                                                    self.max_value,
                                                    self.min_value, with_amplitude, self.amplitude, self.amplitude_2,
                                                    with_damping,
                                                    self.damping_coeff_peaks, self.damping_coeff_valleys)
            self.PLOT_MANAGER.save_plot(fig,
                                        f"{self.measurement.file_name}_{self.measurement.id}_{self.sensor_name}",
                                        subdir="oscillation_1")
            logger.info(f"plot_oscillation for measurement: '{self}' successful.")
        except Exception as e:
            logger.error(f"Failed to plot_oscillation: '{self}'. Error: {e}")

    def fit_and_plot(self, initial_params: Dict[str, float], param_bounds: Dict[str, float],
                     metrics_warning: Dict[str, Tuple[Optional[float], Optional[float]]] = None):
        try:
            param_labels = list(initial_params.keys())

            initial_params_list = convert_dict_to_list(initial_params)
            lower_bounds, upper_bounds = zip(*convert_dict_to_list(param_bounds))
            param_bounds_list = (lower_bounds, upper_bounds)

            sensor_name = self.sensor_name
            data_orig = self.df
            data = data_orig.copy()
            data = clean_peaks_and_valleys(data, sensor_name, self.peaks, self.valleys)
            data = interpolate_points(data, sensor_name, 50)
            data = remove_values_above_percentage(data, sensor_name, self.amplitude_2, 200)

            params_optimal = fit_damped_osc(data, sensor_name, initial_params=initial_params_list,
                                            param_bounds=param_bounds_list)

            metrics_dict = calc_metrics(data, sensor_name, params_optimal=params_optimal)

            if metrics_warning is not None:
                # Überprüft jedes Metrik-Element
                for metric, (lower_threshold, upper_threshold) in metrics_warning.items():
                    metric_value = metrics_dict.get(metric, 0)
                    if (lower_threshold is not None and metric_value < lower_threshold) or \
                            (upper_threshold is not None and metric_value > upper_threshold):
                        logger.warning(
                            f"{self.measurement.id}_{self.sensor_name}: '{metric}' metric value {metric_value} outside of threshold range {lower_threshold}-{upper_threshold}")
            # ...

            fig = plot_oscillation.plot_osc_fit(data, data_orig, sensor_name, param_labels=param_labels,
                                                params_optimal=params_optimal, metrics=metrics_dict,
                                                peaks=self.peaks,
                                                valleys=self.valleys)

            self.PLOT_MANAGER.save_plot(fig,
                                        f"{self.measurement.id}_{self.sensor_name}_{self.measurement.file_name}",
                                        subdir="oscillation_fit_1")
            logger.info(f"fit_and_plot for measurement: '{self}' successful.")
            return metrics_dict


        except Exception as e:
            logger.error(f"Failed to fit_and_plot: '{self}'. Error: {e}")

            return None

    def correct_osc(self):
        pass

    def fit_osc(self, initial_params: Dict[str, float], param_bounds: Dict[str, float],
                metrics_warning: Dict[str, Tuple[Optional[float], Optional[float]]] = None):

        pass

    def plot_osc(self):
        pass
