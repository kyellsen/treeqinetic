import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.signal import find_peaks


from .base_class import BaseClass

from ..plotting import plot_oscillation as plt_o
from ..analyse.fitting_functions import exp_decreasing, damped_oscillation

from kj_core import get_logger

logger = get_logger(__name__)


class Oscillation(BaseClass):
    def __init__(self, measurement, sensor_name, start_index, df: pd.DataFrame):
        super().__init__()

        self.measurement = measurement
        self.sensor_name = sensor_name  # = Column Name
        self.start_index = start_index
        self.df_orig = df
        self.df = df.reset_index(drop=True)

        # Initialize attributes with placeholder values
        self.offset = None
        self.sample_rate = None
        self.max = None
        self.min = None
        self.idxmax = None
        self.idxmin = None

        self.peaks = []
        self.valleys = []

        self.amplitude = None
        self.amplitude_2 = None

        self.frequency = None

        self.damping_coeff_peaks = None
        self.damping_coeff_valleys = None
        self.damping_coeff_avg = None

        # self.damping_coeff_avg_2 = None
        self.fit_data = None

        # Perform calculations
        self.calculate_offset()

        self.calculate_sample_rate()

        self.calculate_min_max()
        self.calculate_peaks_and_valleys()
        self.calculate_amplitude()
        self.calculate_amplitude_2()
        self.calculate_frequency()

        self.calculate_damping_coefficient()
        # self.plot_oscillation_with_damping()

        # self.fit_damped_oscillation()
        # self.plot_oscillation_fitting()

    def __str__(self):
        return f"Oscillation: '{self.measurement.file_name}', ID: {self.measurement.id}, Sensor: {self.sensor_name}'"

    def calculate_offset(self, window_size=5):
        last_quarter = int(0.75 * len(self.df))
        last_data = self.df[self.sensor_name][last_quarter:]
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
            self.max = self.df[self.sensor_name].max()
            self.min = self.df[self.sensor_name].min()
            self.idxmax = self.df[self.sensor_name].idxmax()
            self.idxmin = self.df[self.sensor_name].idxmin()
            logger.debug(f"Calculated min, max, idxmin, and idxmax for sensor {self.sensor_name}.")
        except Exception as e:
            logger.error(f"Failed to calculate min, max, idxmin, and idxmax for sensor {self.sensor_name}: {e}")

    def calculate_amplitude(self):
        try:
            self.amplitude = (self.max - self.min) / 2
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
        self.peaks.insert(0, {'index': 0, 'time': self.df['Sec_Since_Start'].iloc[0], 'value': self.max})

    def calculate_amplitude_2(self):
        try:
            self.amplitude_2 = (self.peaks[1]['value'] - self.min) / 2
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

    # def fit_damped_oscillation(self, maxfev=2000):
    #     try:
    #         # Extract time and amplitude data from the DataFrame
    #         time = self.df['Sec_Since_Start'].values
    #         amplitude_data = self.df[self.sensor_name].values
    #
    #         # Initial parameter guesses
    #         initial_amplitude = self.amplitude_2
    #         damping_coeff = 0.05# self.damping_coeff_avg if self.damping_coeff_avg else 0.25
    #         angular_frequency = self.frequency #2 * np.pi * self.frequency if self.frequency else 1
    #         phase_angle = 0
    #         # Perform the curve fitting
    #         params, _ = curve_fit(damped_oscillation, time, amplitude_data,
    #                               p0=[initial_amplitude, damping_coeff, angular_frequency, phase_angle], maxfev=maxfev)
    #
    #         # Update the object's attributes with the fitted parameters
    #         self.damping_coeff_avg_2 = params[1]
    #         self.fit_data = damped_oscillation(time, *params)
    #
    #         logger.info(f"Damped oscillation fitting successful for sensor {self.sensor_name}.")
    #
    #     except Exception as e:
    #         logger.error(f"Failed to fit damped oscillation for sensor {self.sensor_name}: {e}")
    #         self.damping_coeff_avg_2 = None
    #         self.fit_data = None

    def plot_oscillation(self):
        """

        """
        try:
            plt_o.plot_oscillation(self)
            logger.info(f"plot_oscillation for measurement: '{self}' successful.")
        except Exception as e:
            logger.error(f"Failed to plot_oscillation: '{self}'. Error: {e}")

    def plot_oscillation_with_damping(self):
        """

        """
        try:
            plt_o.plot_oscillation_with_damping(self)
            logger.info(f"plot_oscillation_with_damping for measurement: '{self}' successful.")
        except Exception as e:
            logger.error(f"Failed to plot_oscillation_with_damping: '{self}'. Error: {e}")

    # def plot_oscillation_fitting(self):
    #     """
    #
    #     """
    #     try:
    #         plt_o.plot_oscillation_fitting(self)
    #         logger.info(f"plot_oscillation_fitting for measurement: '{self}' successful.")
    #     except Exception as e:
    #         logger.error(f"Failed to plot_oscillation_fitting: '{self}'. Error: {e}")
