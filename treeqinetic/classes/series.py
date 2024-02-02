from pathlib import Path
import numpy as np
import pandas as pd
from typing import List

from .base_class import BaseClass
from .measurement import Measurement

from ..plotting.plot import plot_error_histogram, plot_error_qq, plot_error_violin

from kj_logger import get_logger

logger = get_logger(__name__)


class Series(BaseClass):
    def __init__(self, name: str, path: str):
        super().__init__()
        Measurement.counter = 0  # reset the counter for the Measurement class
        self.name = name
        self.path = Path(path)
        self.measurement_files_paths = [f for f in self.path.iterdir() if f.is_file() and f.suffix == '.txt']
        self.measurement_files = [f.name for f in self.path.iterdir() if f.is_file() and f.suffix == '.txt']
        self.measurements = []

        for measurement_file_path in self.measurement_files_paths:
            if measurement_file_path.name not in self.measurements:
                # Create an instance of Measurement and add it to the list "measurements"
                self.measurements.append(Measurement.read_txt(file_path=measurement_file_path))

        # Set the attribute "measurements_count" to the number of measurements in the list "measurements"
        self.measurements_count = len(self.measurements)

        # Create df_list and df for all Logs of the series at initialization
        self.df_list = self.get_measurements_df_list()
        self.df = self.get_measurements_df()

        self.osc_df = None
        self.osc_errors_dict_all = None
        self.osc_errors_dict_by_sensor = None

    def __str__(self):
        return f"Series: '{self.name}' with {self.measurements_count} measurements: {self.measurement_files}"

    def get_measurements_df_list(self):
        return [measurement.data for measurement in self.measurements]

    def get_measurements_df(self):
        return pd.concat(self.df_list, ignore_index=True)

    def plot_measurement_sensors(self, sensor_names: list, time_start=None, time_end=None):
        for measurement in self.measurements:
            measurement.plot_multi_sensors(sensor_names, time_start, time_end)

    def get_oscillations(self, sensor_names: list):
        for measurement in self.measurements:
            logger.info(f"\n Select Oscillations for {measurement}")
            measurement.select_oscillations(sensor_names)

    def plot_oscillations_for_measurements(self, sensor_names: list, combined: bool = False):
        for measurement in self.measurements:
            logger.info(f"Plot Oscillations for {measurement}")
            measurement.plot_select_oscillations(sensor_names, combined)

    def get_oscillations_list(self):
        oscillation_list = []
        for measurement in self.measurements:
            oscillation_list.extend(measurement.oscillations.values())
        return oscillation_list

    def get_oscillations_df(self):
        oscillations = self.get_oscillations_list()
        data = []

        for osc in oscillations:
            row = {
                'id': osc.measurement.cms_id,
                'file_name': osc.measurement.file_name,
                'sensor_name': osc.sensor_name,
                'sample_rate': osc.sample_rate,
                'm_y_shift': osc.m_y_shift,
                'max_value': osc.max_value,
                'min_value': osc.min_value,
                'm_amplitude': osc.m_amplitude,
                'm_amplitude_2': osc.m_amplitude_2,
                'm_frequency': osc.m_frequency,
                'initial_amplitude': osc.initial_amplitude,
                'damping_coeff': osc.damping_coeff,
                'angular_frequency': osc.angular_frequency,
                'phase_angle': osc.phase_angle,
                'y_shift': osc.y_shift,
                'metrics_warning': osc.metric_warning,
                'mse': osc.mse,
                'mae': osc.mae,
                'rmse': osc.rmse,
                'r2': osc.r2,
            }
            data.append(row)
        df = pd.DataFrame(data)

        return df

    @staticmethod
    def normalize_errors(errors: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Normalizes the errors based on the provided scale factor.

        Args:
            errors (np.ndarray): The array of errors to be normalized.
            scale_factor (float): The scale factor used for normalization.

        Returns:
            np.ndarray: Normalized errors.
        """
        return errors / scale_factor if scale_factor != 0 else errors

    def collect_and_normalize_errors(self, oscillations: List, sensor_name: str = None) -> np.ndarray:
        """
        Collects and normalizes errors for a specific sensor or all sensors combined.

        Args:
            oscillations (List): List of oscillation objects.
            sensor_name (str, optional): Specific sensor name to collect errors for. If None, errors for all sensors are collected.

        Returns:
            np.ndarray: Normalized errors.
        """
        if sensor_name:
            sensor_data = [osc.df_fit[sensor_name] for osc in oscillations if osc.sensor_name == sensor_name]
            sensor_errors = [osc.errors for osc in oscillations if osc.sensor_name == sensor_name]
        else:
            sensor_data = [np.array(osc.df_fit[osc.sensor_name]) for osc in oscillations]
            sensor_errors = [osc.errors for osc in oscillations]

        scale_factor = max(np.max(np.abs(data)) for data in sensor_data)
        normalized_errors = np.concatenate(
            [self.normalize_errors(errors, scale_factor) for errors in sensor_errors])

        return normalized_errors

    def get_osc_errors(self) -> None:
        """
        Collects and normalizes errors for each sensor and combined for all sensors.

        This method updates the class attributes osc_errors_dict_by_sensor and osc_errors_dict_all.
        """
        oscillations = self.get_oscillations_list()
        self.osc_errors_dict_by_sensor = {sensor_name: self.collect_and_normalize_errors(oscillations, sensor_name)
                                          for sensor_name in set(osc.sensor_name for osc in oscillations)}
        self.osc_errors_dict_all = {"all_fits": self.collect_and_normalize_errors(oscillations)}

    def plot_osc_errors(self, plot_qq: bool = True, plot_violin: bool = True, hist_trim_percent: float = None,
                        plot_hist: bool = True):
        """
        Plots the normalized errors for each sensor and combined for all sensors.

        Args:
            plot_qq (bool): If True, plots a QQ plot of the normalized errors.
            plot_violin
            plot_hist (bool): If True, plots a histogram of the normalized errors.
            hist_trim_percent (float): Percentage of data to trim from each end for the histogram.

        """

        # Stellen Sie sicher, dass die Fehlerdaten verfügbar sind
        self.get_osc_errors()

        # Plotting für alle Sensoren zusammen
        if plot_hist or plot_qq:
            self._plot_error_data(self.osc_errors_dict_all["all_fits"], "all_fits",
                                  plot_qq, plot_violin, plot_hist, hist_trim_percent)

        # Plotting für jeden Sensor separat
        for sensor_name, errors in self.osc_errors_dict_by_sensor.items():
            self._plot_error_data(errors, sensor_name, plot_qq, False, plot_hist, hist_trim_percent)

    def _plot_error_data(self, errors, title_suffix, plot_qq, plot_violin, plot_hist, hist_trim_percent,
                         sub_dir="series_osc_errors"):
        """
        Helper function to plot error data.

        Args:
            errors (np.ndarray): Array of errors to plot.
            title_suffix (str): Suffix to append to the plot title.
            plot_hist (bool): If True, plots a histogram of the errors.
            hist_trim_percent (float): Percentage of data to trim from histogram.
            plot_qq (bool): If True, plots a QQ plot of the errors.
        """
        if plot_hist:
            try:
                fig = plot_error_histogram(errors,
                                           title=f"Histogram of Normalized Errors for {title_suffix}",
                                           trim_percent=hist_trim_percent)
                self.PLOT_MANAGER.save_plot(fig, f"normalized_errors_hist_{title_suffix}",
                                            subdir=sub_dir)
                logger.debug(f"plot_error_histogram for {title_suffix}: successful.")
            except Exception as plot_error:
                logger.error(f"Error in plot_error_histogram for {title_suffix}: {plot_error}")

        if plot_qq:
            try:
                fig = plot_error_qq(errors,
                                    title=f"QQ Plot of Normalized Errors for {title_suffix}")
                self.PLOT_MANAGER.save_plot(fig, f"normalized_errors_qq_{title_suffix}",
                                            subdir=sub_dir)
                logger.debug(f"plot_error_qq for {title_suffix}: successful.")
            except Exception as plot_error:
                logger.error(f"Error in plot_error_qq for {title_suffix}: {plot_error}")

        if plot_violin:
            try:
                fig = plot_error_violin(self.osc_errors_dict_by_sensor,
                                        title=f"Violin Plot of Normalized Errors for {title_suffix}")
                self.PLOT_MANAGER.save_plot(fig, f"normalized_errors_violin_{title_suffix}",
                                            subdir=sub_dir)
                logger.debug(f"plot_error_violin for {title_suffix}: successful.")
            except Exception as plot_error:
                logger.error(f"Error in plot_error_violin for {title_suffix}: {plot_error}")
