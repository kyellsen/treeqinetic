import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from scipy.signal import find_peaks

from .base_class import BaseClass

from ..plotting.plot import plot_error_histogram
from ..plotting import plot_oscillation
from kj_core.df_utils.df_calc import calc_sample_rate, calc_amplitude, calc_min_max
from ..analyse.correct_oscillation import zero_base_column, remove_values_above_percentage, clean_peaks_and_valleys, \
    interpolate_points
from ..analyse.fitting_functions import damped_osc, fit_damped_osc_mae, calc_metrics

from kj_core import get_logger

logger = get_logger(__name__)


class Oscillation(BaseClass):

    def __init__(self, measurement, sensor_name, start_index, df: pd.DataFrame):
        super().__init__()

        self.measurement = measurement
        self.sensor_name = sensor_name  # = Column Name
        self.start_index = start_index
        self.df_orig = df
        self.df_orig_max = df[sensor_name].max()
        self.df_orig_min = df[sensor_name].min()

        # Reset the index of the DataFrame and make sure we don't lose the original index
        self.df = df.reset_index(drop=True, inplace=False)  # inplace=False -> creates a copy of df
        # Normalize the 'Sec_Since_Start' column by subtracting its minimum value
        self.df = zero_base_column(self.df, "Sec_Since_Start")

        self.df_fit = None

        # Initialize attributes with placeholder values
        self.sample_rate = None

        self.max_idx = None
        self.max_time = None
        self.max_value = None

        self.min_idx = None
        self.min_time = None
        self.min_value = None

        self.peaks = []
        self.valleys = []

        # param from manuell fitting
        self.m_amplitude = None
        self.m_amplitude_2 = None
        self.m_frequency = None
        self.m_y_shift = None

        # param from automatic fitting
        self.param_optimal = None
        self.param_optimal_dict = None

        self.initial_amplitude = None
        self.damping_coeff = None
        self.angular_frequency = None
        self.phase_angle = None
        self.y_shift = None

        # Quality metrics from fitting
        self.metrics_dict = None
        self.metric_warning = False

        self.mse = None
        self.mae = None
        self.rmse = None
        self.r2 = None

        self.errors = None

        # Perform manuel calculations
        self.calc_m_y_shift()

        self.sample_rate = calc_sample_rate(self.df, "Sec_Since_Start")
        self.m_amplitude = calc_amplitude(self.df, self.sensor_name)
        self.calc_min_max()
        self.calc_peaks_and_valleys()
        self.calc_m_amplitude_2()
        self.calc_m_frequency()

    def __str__(self):
        return f"Oscillation: '{self.measurement.file_name}', ID: {self.measurement.id}, Sensor: {self.sensor_name}'"

    def calc_m_y_shift(self, window_size: int = 5, last_percent: float = 60) -> float:
        """
        Calculates and applies a y_shift based on the rolling mean of the last percentage of data.

        This method modifies the sensor data by subtracting the calculated y_shift.

        Args:
            window_size (int): The size of the moving window for calculating the mean. Default is 5.
            last_percent (float): The percentage of the latest data points to consider for the y_shift calculation. Default is 60.

        Returns:
            float: The calculated y_shift value.
        """
        # Calculate the number of points for the specified percentage of the data
        last_points = int(last_percent / 100 * len(self.df))
        # Retrieve the last part of the data for the specified sensor
        last_data = self.df[self.sensor_name][last_points:]
        # Calculate the y_shift as the mean of the rolling mean
        y_shift = last_data.rolling(window=window_size).mean().dropna().mean()
        # Apply the y_shift to the sensor data
        # self.df[self.sensor_name] -= y_shift
        # Store and return the y_shift
        self.m_y_shift = y_shift
        return y_shift

    def calc_min_max(self) -> dict:
        """
        Calculates and stores the minimum and maximum values, their corresponding indices, and times for the specified sensor.

        This method utilizes the 'calc_min_max' function to perform the calculation.

        Returns:
            dict: A dictionary containing 'max' and 'min' information, each with 'idx', 'time', and 'value'.

        Raises:
            RuntimeError: If the calculation fails.
        """
        try:
            min_max_dict = calc_min_max(self.df, self.sensor_name, 'Sec_Since_Start')
            self.max_idx = min_max_dict['max']['idx']
            self.max_time = min_max_dict['max']['time']
            self.max_value = min_max_dict['max']['value']
            self.min_idx = min_max_dict['min']['idx']
            self.min_time = min_max_dict['min']['time']
            self.min_value = min_max_dict['min']['value']

            logger.debug(f"Calculated min and max (idx, time and value) for sensor {self.sensor_name}.")
            return min_max_dict

        except Exception as e:
            logger.error(f"Failed to calculate min and max (idx, time and value) for sensor {self.sensor_name}: {e}")
            raise RuntimeError(f"Failed to calculate min and max: {e}")

    def calc_peaks_and_valleys(self, min_distance_sec: float = 1.0, search_range: float = 0.75,
                               prominence: float = 5.0) -> Dict:
        """
        Identifies peaks and valleys in the sensor data within a specified search range.

        Args:
            min_distance_sec (float): Minimum distance between peaks/valleys in seconds. Default is 1.0 second.
            search_range (float): Proportion of the data to search for peaks/valleys, given as a fraction (0 to 1). Default is 0.75 (75%).
            prominence (float): Required prominence of peaks/valleys. Default is 5.

        Updates:
            self.peaks (list): List of dictionaries containing peak information ('index', 'time', and 'value').
            self.valleys (list): Similar to self.peaks, but for valleys.
        """
        min_distance_samples = float(min_distance_sec * self.sample_rate)
        cut_idx = int(len(self.df) * search_range)

        idx_peaks, _ = find_peaks(self.df[self.sensor_name][:cut_idx], distance=min_distance_samples,
                                  prominence=prominence)
        idx_valleys, _ = find_peaks(-self.df[self.sensor_name][:cut_idx], distance=min_distance_samples,
                                    prominence=prominence)

        self.peaks = [
            {'index': idx, 'time': self.df['Sec_Since_Start'].iloc[idx], 'value': self.df[self.sensor_name].iloc[idx]}
            for idx in idx_peaks]
        self.valleys = [
            {'index': idx, 'time': self.df['Sec_Since_Start'].iloc[idx], 'value': self.df[self.sensor_name].iloc[idx]}
            for idx in idx_valleys]

        if len(self.peaks) > 1 and self.peaks[0]['value'] < self.peaks[1]['value']:
            del self.peaks[0]

        self.peaks.insert(0, {'index': 0, 'time': self.df['Sec_Since_Start'].iloc[0],
                              'value': self.df[self.sensor_name].max()})

        return {"peaks:": self.peaks, "valleys": self.valleys}

    def calc_m_amplitude_2(self):
        try:
            self.m_amplitude_2 = (self.peaks[1]['value'] - self.min_value) / 2
            logger.debug(f"Calculated amplitude_2 for sensor {self.sensor_name}.")

        except Exception as e:
            self.m_amplitude_2 = self.m_amplitude / 2
            logger.warning(
                f"Failed to calculate amplitude_2 for sensor {self.sensor_name}. Using half of amplitude as fallback. Error: {e}")
        return self.m_amplitude_2

    def calc_m_frequency(self) -> Optional[float]:
        """
        Calculates the mean frequency based on the time intervals between peaks and valleys.

        This method extracts time values from the 'peaks' and 'valleys' attributes of the class,
        computes the time differences between consecutive peaks and valleys, and then calculates
        the mean of these time differences. The inverse of this mean time delta is considered as
        the mean frequency, which is then set to the 'm_frequency' attribute.

        Returns:
            The calculated mean frequency as a float, if successful. Returns None if the calculation
            cannot be completed due to data issues.
        """

        try:
            # Extracting time values for peaks and valleys
            peak_times = np.array([peak['time'] for peak in self.peaks])
            valley_times = np.array([valley['time'] for valley in self.valleys])

            # Ensure there are enough data points to calculate differences
            if len(peak_times) < 2 or len(valley_times) < 2:
                logger.warning("Insufficient data points in peaks or valleys for frequency calculation.")
                return None

            # Calculating time deltas
            time_deltas_peaks = np.diff(peak_times)
            time_deltas_valleys = np.diff(valley_times)

            # Concatenating and computing mean
            mean_time_delta = np.mean(np.concatenate((time_deltas_peaks, time_deltas_valleys)))

            # Prevent division by zero
            if mean_time_delta == 0:
                logger.error("Mean time delta is zero, cannot calculate frequency.")
                return None

            # Calculating frequency
            self.m_frequency = 1 / mean_time_delta
            return self.m_frequency

        except Exception as e:
            logger.critical(f"Unexpected error occurred during frequency calculation: {e}", exc_info=True)
            return None

    def get_initial_param(self):
        initial_param_labels = self.CONFIG.Oscillation.param_labels
        initial_param_values = self.CONFIG.Oscillation.initial_param_values

        initial_param_dict = {label: value for label, value in zip(initial_param_labels, initial_param_values)}

        return initial_param_dict

    def get_param_bounds(self):
        initial_param_labels = self.CONFIG.Oscillation.param_labels
        bounds_values = self.CONFIG.Oscillation.bounds_values
        param_bounds_dict = {label: bounds for label, bounds in zip(initial_param_labels, bounds_values)}

        return param_bounds_dict

    def get_metrics_warning(self):
        metrics_labels = self.CONFIG.Oscillation.metrics_labels
        metrics_warning_values = self.CONFIG.Oscillation.metrics_warning_values
        metrics_warning_dict = {label: value for label, value in zip(metrics_labels, metrics_warning_values)}

        return metrics_warning_dict

    def fit(self, initial_param: Dict[str, float] = None, param_bounds: Dict[str, Tuple] = None,
            metrics_warning: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
            plot: bool = True, plot_error: bool = False, dir_add: str = None, clean_peaks=True, interpolate=True,
            remove_values_above=True) -> None:
        """
        Fits the model to the data using the provided initial parameters and parameter bounds.
        It calculates optimal parameters, metrics, and optionally plots the results.

        The function involves several steps:
        1. Preparing the data.
        2. Calculating optimal parameters.
        3. Computing metrics based on the optimal parameters.
        4. Optionally plotting the results if `plot` is True.

        Parameters:
        - initial_param: A dictionary of initial parameter values.
        - param_bounds: A dictionary of tuples representing the lower and upper bounds for each parameter.
        - metrics_warning: An optional dictionary specifying warning thresholds for each metric.
        - plot: A boolean flag to indicate whether to generate a plot.
        - dir_add

        Returns:
        - None
        """
        if initial_param is None:
            initial_param = self.get_initial_param()
        if param_bounds is None:
            param_bounds = self.get_param_bounds()
        if metrics_warning is None:
            metrics_warning = self.get_metrics_warning()

        try:
            self.get_df_fit(clean_peaks, interpolate, remove_values_above)
            self.calc_param_optimal(initial_param, param_bounds)
            self.calc_metrics(metrics_warning)
            self.calc_errors()

            self.set_fit_attributes()
            logger.info(f"fit for measurement: '{self}' successful.")

            if plot:
                self.plot_fit(dir_add)
            if plot_error:
                self.plot_fit_errors(dir_add)

        except Exception as e:
            logger.critical(f"Error in fit method: {e}", exc_info=True)

    def get_df_fit(self, clean_peaks, interpolate, remove_values_above) -> None:
        """
        Prepares the dataframe for fitting by cleaning and interpolating data.

        This method performs the following operations on the dataframe:
        1. Copies the original dataframe.
        2. Cleans peaks and valleys.
        3. Interpolates points.
        4. Removes values above a certain percentage threshold.

        Returns:
        - None
        """
        try:
            df_fit = self.df.copy()
            if clean_peaks:
                df_fit = clean_peaks_and_valleys(df_fit, self.sensor_name, self.peaks, self.valleys)
            if interpolate:
                df_fit = interpolate_points(df_fit, self.sensor_name, 50)
            if remove_values_above:
                df_fit = remove_values_above_percentage(df_fit, self.sensor_name, self.m_amplitude_2, 200)
            self.df_fit = df_fit
        except Exception as e:
            logger.error(f"Error in get_df_fit: {e}")

    def calc_param_optimal(self, initial_param: Dict[str, float],
                           param_bounds: Dict[str, Tuple[float, float]]) -> None:
        """
        Calculates the optimal parameters for the model.

        Parameters:
        - initial_param: A dictionary of initial parameter values.
        - param_bounds: A dictionary of tuples representing the lower and upper bounds for each parameter.

        Returns:
        - None
        """
        try:
            initial_param_list = [initial_param[key] for key in initial_param]
            lower_bounds, upper_bounds = zip(*[param_bounds[key] for key in param_bounds])
            param_bounds_list = (lower_bounds, upper_bounds)
            self.param_optimal = fit_damped_osc_mae(self.df_fit, self.sensor_name, initial_param=initial_param_list, param_bounds=param_bounds_list)
            #self.param_optimal = fit_damped_osc(self.df_fit, self.sensor_name, initial_param=initial_param_list,
                                                    #param_bounds=param_bounds_list)
            param_labels = self.CONFIG.Oscillation.param_labels
            self.param_optimal_dict = {label: param for label, param in zip(param_labels, self.param_optimal)}
        except Exception as e:
            logger.error(f"Error in calc_param_optimal: {e}")

    def calc_metrics(self, metrics_warning: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]]) -> \
            Dict[str, float]:
        """
        Calculates and evaluates metrics based on the fitted model parameters.

        This method computes various metrics to assess the quality of the fit and checks them against
        optional warning thresholds. If a metric falls outside its specified threshold range, a warning
        is logged.

        Args:
            metrics_warning: An optional dictionary where each key is a metric name, and its value is a
                             tuple (lower_threshold, upper_threshold). If a metric is outside this range,
                             a warning is logged.

        Returns:
            A dictionary of calculated metrics.
        """

        try:
            metrics_values = calc_metrics(self.df_fit, self.sensor_name, self.param_optimal)
        except Exception as e:
            logger.error(f"Error in calculating metrics: {e}", exc_info=True)
            return {}

        metrics_labels = self.CONFIG.Oscillation.metrics_labels
        metrics_dict = {label: value for label, value in zip(metrics_labels, metrics_values)}

        for metric, (lower_threshold, upper_threshold) in metrics_warning.items():
            metric_value = metrics_dict.get(metric, 0)
            if (lower_threshold is not None and metric_value < lower_threshold) or \
                    (upper_threshold is not None and metric_value > upper_threshold):
                logger.warning(
                    f"{self.measurement.id}_{self.sensor_name}: '{metric}' metric value {metric_value} "
                    f"outside of threshold range {lower_threshold}-{upper_threshold}")
                self.metric_warning = True

        self.metrics_dict = metrics_dict
        return metrics_dict

    # Funktion zur Berechnung der Fehler
    def calc_errors(self) -> np.ndarray:
        """
        Calculates the errors (residuals) between the observed data and the model predictions.

        Args:
        data (pd.DataFrame): DataFrame containing the original data.
        sensor_name (str): Column name of the sensor data.
        param_optimal (np.ndarray): Optimal parameters from the curve fitting.

        Returns:
        np.ndarray: Array of residuals/errors for each data point.
        """
        fitted_values = damped_osc(self.df_fit['Sec_Since_Start'], *self.param_optimal)
        errors = self.df_fit[self.sensor_name] - fitted_values
        self.errors = errors
        return errors

    def plot_fit(self, dir_add: Optional[str] = None) -> None:
        """
        Plots the fitting results and saves the plot.

        Parameters:
        - dir_add: Optional sub-directory name for saving the plot.
        """
        try:
            fig = plot_oscillation.plot_osc_fit(self.df_fit, self.df, self.sensor_name,
                                                self.param_optimal_dict, self.param_optimal,
                                                self.metrics_dict, metrics_warning=self.metric_warning,
                                                peaks=self.peaks, valleys=self.valleys)

            self.PLOT_MANAGER.save_plot(fig, f"{self.measurement.id}_{self.sensor_name}_{self.measurement.file_name}",
                                        subdir=f"osc_fit_1{dir_add}")
            logger.debug(f"Plot for measurement: '{self}' successful.")
        except Exception as plot_error:
            logger.error(f"Error in plotting: {plot_error}")

    def plot_fit_errors(self, dir_add: Optional[str] = None) -> None:
        """
        Plots the fitting results and saves the plot.

        Parameters:
        - dir_add: Optional sub-directory name for saving the plot.
        """
        try:
            fig = plot_error_histogram(self.errors, show_plot=False)

            self.PLOT_MANAGER.save_plot(fig, f"{self.measurement.id}_{self.sensor_name}_{self.measurement.file_name}",
                                        subdir=f"osc_fit_errors_1{dir_add}")
            logger.debug(f"Plot for measurement: '{self}' successful.")
        except Exception as e:
            logger.error(f"Error in plotting: {e}")

    def set_fit_attributes(self):
        """
        Setzt die Instanzattribute basierend auf den Ergebnissen des Fit-Prozesses.

        Diese Methode aktualisiert die Instanzattribute mit den Werten aus
        `self.param_optimal_dict` und `self.metrics_dict`, die nach dem Fit-Prozess berechnet wurden.
        """

        # Aktualisieren der Parameter-Attribute, wenn sie vorhanden sind
        if self.param_optimal_dict is not None:
            self.initial_amplitude = self.param_optimal_dict.get("initial_amplitude")
            self.damping_coeff = self.param_optimal_dict.get("damping_coeff")
            self.angular_frequency = self.param_optimal_dict.get("angular_frequency")
            self.phase_angle = self.param_optimal_dict.get("phase_angle")
            self.y_shift = self.param_optimal_dict.get("y_shift")

        # Aktualisieren der Metrik-Attribute, wenn sie vorhanden sind
        if self.metrics_dict is not None:
            self.mse = self.metrics_dict.get("mse")
            self.mae = self.metrics_dict.get("mae")
            self.rmse = self.metrics_dict.get("rmse")
            self.r2 = self.metrics_dict.get("r2")
