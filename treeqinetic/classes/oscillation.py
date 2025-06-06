import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List

from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator

from .base_class import BaseClass

from ..plotting.plot_fit_error import plot_error_histogram
from ..plotting import plot_oscillation
from ..plotting.plot_integral import plot_integral

from kj_core.df_utils.df_calc import calc_sample_rate, calc_amplitude, calc_min_max
from kj_core.classes.similarity_metrics import SimilarityMetrics
from ..analyse.fitting_functions import damped_osc, fit_damped_osc
from ..analyse.calc_param_add import calc_frequency_undamped, calc_damping_ratio

from kj_logger import get_logger

logger = get_logger(__name__)


class Oscillation(BaseClass):

    def __init__(self, measurement, sensor_name: str, start_index: int, df: pd.DataFrame):
        super().__init__()
        self.measurement = measurement
        self.sensor_name = sensor_name
        self.start_index = start_index
        self.oscillation_df = df    # from Measurement.select_oscillations

        # Prepare the DataFrame
        self.df = self._prepare_dataframe(df)   # data index / time reset to 0

        self.df_fit = None
        self.sample_rate = None
        self.max_idx = None
        self.max_time = None
        self.max_value = None
        self.min_idx = None
        self.min_time = None
        self.min_value = None
        self.peaks = []
        self.valleys = []
        self.m_amplitude = None
        self.m_amplitude_2 = None
        self.param_optimal = None
        self.param_optimal_dict = None
        self.optimization_details = None
        self.metrics = None
        self.metrics_dict = None
        self.metric_warning = False
        self.errors = None

        self.integral_dict = None

        # Perform initial calculations
        self._initial_calculations()

    def __str__(self):
        return f"Oscillation: '{self.measurement.file_name}', ID: {self.measurement.id}, Sensor: {self.sensor_name}'"

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and normalize the DataFrame.
        """
        df = df.reset_index(drop=True, inplace=False)
        if "Sec_Since_Start" not in df.columns:
            raise ValueError("Column 'Sec_Since_Start' not found in DataFrame.")
        min_value = df["Sec_Since_Start"].min()
        df["Sec_Since_Start"] -= min_value
        return df

    def _initial_calculations(self) -> None:
        """
        Perform initial calculations including sample rate, amplitude, min/max values, peaks, and valleys.
        """
        self.sample_rate = calc_sample_rate(self.df, "Sec_Since_Start")
        self.m_amplitude = calc_amplitude(self.df, self.sensor_name)
        self.calc_min_max()
        self.calc_peaks_and_valleys()
        self.calc_m_amplitude_2()

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

    def fit(self, initial_param: Dict[str, float] = None, param_bounds: Dict[str, Tuple] = None, optimize_criterion: str = None,
            metrics_warning: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = None,
            plot: bool = True, plot_error: bool = False, dir_add: str = None, interpolate=True) -> None:
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
        - optimize_criterion (str): Criterion to optimize ('mae' or 'mse').
        - metrics_warning: An optional dictionary specifying warning thresholds for each metric.
        - plot: A boolean flag to indicate whether to generate a plot.
        - plot_error: A boolean flag to indicate whether to generate a error plot.
        - dir_add
        - interpolate

        Returns:
        - None
        """
        if initial_param is None:
            initial_param = self.CONFIG.Oscillation.initial_param
        if param_bounds is None:
            param_bounds = self.CONFIG.Oscillation.param_bounds
        if metrics_warning is None:
            metrics_warning = self.CONFIG.Oscillation.metrics_warning

        try:
            if self.df_fit is None:
                self.prepare_df_for_fitting(interpolate)
            self.calc_param_optimal(initial_param, param_bounds, optimize_criterion)
            self.calc_metrics(metrics_warning)
            self.calc_errors()

            logger.info(f"fit for measurement: '{self}' successful.")

            if plot:
                self.plot_fit(dir_add)
            if plot_error:
                self.plot_fit_errors(dir_add)
        except Exception as e:
            logger.critical(f"Error in fit method: {e}", exc_info=True)

    def prepare_df_for_fitting(self, interpolate: bool, sample_rate: float = 50.0) -> None:
        """
        Prepares the dataframe for fitting by cleaning and interpolating data.

        This method performs the following operations on the dataframe:
        1. Copies the original dataframe.
        2. Interpolates points if requested.

        Args:
        interpolate (bool): Whether to interpolate points.
        sample_rate (float): Desired sample rate for interpolation in Hz.

        Returns:
        - None
        """
        try:
            df_fit = self.df.copy()
            if interpolate:
                # Integration of interpolation directly within the method
                interpolator = PchipInterpolator(df_fit['Sec_Since_Start'], df_fit[self.sensor_name])
                min_time = df_fit['Sec_Since_Start'].min()
                max_time = df_fit['Sec_Since_Start'].max()
                total_time = max_time - min_time
                num_points = int(total_time * sample_rate)

                new_times = np.linspace(min_time, max_time, num=num_points)
                new_values = interpolator(new_times)

                df_fit = pd.DataFrame({'Sec_Since_Start': new_times, self.sensor_name: new_values})

            self.df_fit = df_fit
        except Exception as e:
            logger.error(f"Error in prepare_df_for_fitting: {e}")

    def calculate_elasto_integral(
            self,
            interpolate: bool = True,
            sample_rate: float = 50.0,
            last_frac: float = 0.50,
            unit: str = "µm·s",
            plot: bool = True
    ) -> None:
        """
        Berechnet das zeitliche Integral der Sensorsignale und speichert die Ergebnisse.

        Parameters
        ----------
        interpolate : bool, optional
            Legt fest, ob die Rohdaten vor der Integration interpoliert werden sollen.
            Standard ist True.
        sample_rate : float, optional
            Abtastrate in Hertz (Hz) zur Bestimmung des Zeitintervalls zwischen den
            Datenpunkten. Muss größer als 0 sein. Standard ist 50.0.
        last_frac : float, optional
            Anteil der letzten Datenpunkte zur Berechnung des Mittelwerts für den
            Intercept (Nullpunkt). Muss im Bereich (0, 1] liegen. Standard ist 0.50.
        unit : str, optional
            Einheit der berechneten Integrale (z. B. "µm·s"). Standard ist "µm·s".
        plot : bool, optional
            Gibt an, ob das Ergebnis geplottet und abgespeichert werden soll.
            Standard ist True.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            Wenn `sample_rate` ≤ 0 oder `last_frac` nicht im (0, 1] liegt.

        Notes
        -----
        1. Lädt bzw. interpoliert die Daten, falls sie noch nicht vorbereitet sind.
        2. Validiert die Eingabeparameter für Abtastrate und letzten Datenanteil.
        3. Bestimmt den Mittelwert der letzten `last_frac` Datenpunkte als Intercept
           und zentriert die Messwerte.
        4. Berechnet das positive und negative Integral getrennt per Summation
           und multipliziert mit dem Zeitintervall.
        5. Speichert die Ergebnisse in `self.integral_dict` und schreibt sie ins Log.
        6. Optional wird ein Plot erzeugt und über `self.PLOT_MANAGER` abgespeichert.
        """
        try:
            # 1) Daten laden/interpolieren
            if self.df_fit is None:
                self.prepare_df_for_fitting(interpolate)

            # 2) Eingaben validieren
            if sample_rate <= 0:
                raise ValueError("sample_rate must be positive")
            if not (0 < last_frac <= 1):
                raise ValueError("last_frac must be in (0, 1]")

            df = self.df_fit.copy()
            sampling_interval = 1.0 / sample_rate

            # 3) Intercept bestimmen
            n_last = max(1, int(last_frac * len(df)))
            intercept = df[self.sensor_name].iloc[-n_last:].mean()

            # 4) Werte zentrieren
            df['cent'] = df[self.sensor_name] - intercept

            # 5) Positive und negative Anteile mit pandas clip statt numpy
            pos = df['cent'].clip(lower=0)
            neg = (-df['cent'].clip(upper=0))

            pos_int = pos.sum() * sampling_interval
            neg_int = neg.sum() * sampling_interval
            abs_int = pos_int + neg_int
            ratio = pos_int / neg_int if neg_int > 0 else float('nan')

            # 6) Ergebnisse speichern
            self.integral_dict = {
                "integral_intercept": intercept,
                "integral_positiv": pos_int,
                "integral_negativ": neg_int,
                "integral_abs": abs_int,
                "integral_ratio": ratio,
                #"integral_unit": unit
            }

            logger.info(
                f"calculate_elasto_integral successful: "
                f"pos={pos_int:.3f}, neg={neg_int:.3f}, abs={abs_int:.3f}, ratio={ratio:.3f}"
            )

            # 7) Optional plotten
            if plot:
                fig = plot_integral(df, self.sensor_name, self.measurement.id, self.integral_dict)
                filename = f"{self.measurement.id}_{self.sensor_name}_{self.measurement.file_name}"
                self.PLOT_MANAGER.save_plot(fig, filename, subdir="osc_integral")

        except Exception as e:
            logger.error(f"Error in calculate_elasto_integral: {e}", exc_info=True)

    def calc_param_optimal(self, initial_param: Dict[str, float],
                           param_bounds: Dict[str, Tuple[float, float]],
                           optimize_criterion: str) -> None:
        """
        Calculates the optimal parameters for the model.

        Parameters:
        - initial_param: A dictionary of initial parameter values.
        - param_bounds: A dictionary of tuples representing the lower and upper bounds for each parameter.
        - optimize_criterion (str): Criterion to optimize ('mae' or 'mse').

        Returns:
        - None
        """
        try:
            param_names = list(initial_param.keys())
            initial_param_list = [initial_param[name] for name in param_names]
            lower_bounds, upper_bounds = zip(*[param_bounds[name] for name in param_names])
            param_bounds_list = (list(lower_bounds), list(upper_bounds))

            # fit_options aus der Config
            fit_options = self.CONFIG.Oscillation.fit_options

            result = fit_damped_osc(
                data=self.df_fit,
                sensor_name=self.sensor_name,
                initial_param=initial_param_list,
                param_bounds=param_bounds_list,
                optimize_criterion=optimize_criterion,
                options=fit_options
            )

            # param_optimal speichert nur die optimierten Parameter
            self.param_optimal = result.x

            # 2) Neue Instanzvariable zur Speicherung von Details des OptimizeResult
            self.optimization_details = {
                "nit": result.nit,
                "nfev": result.nfev,
                "success": result.success,
                "status": result.status,
                "fun": result.fun,
                "message": result.message
            }

            # Kompaktes Logging der wichtigsten Infos
            logger.debug(
                f"Optimization result: "
                f"nit={result.nit}, "
                f"nfev={result.nfev}, "
                f"success={result.success}, "
                f"status={result.status}, "
                f"fun={result.fun}, "
                f"message='{result.message}'"
            )

            # Aus den Parametern ein Dictionary mit sprechenden Keys erstellen
            param_labels = self.CONFIG.Oscillation.param_labels
            self.param_optimal_dict = {
                label: param for label, param in zip(param_labels, self.param_optimal)
            }

            # Ableitungen berechnen (frequency_undamped, damping_ratio, usw.)
            self.calc_additional_parameters()

        except Exception as e:
            logger.error(f"Error in calc_param_optimal: {e}")

    def calc_additional_parameters(self) -> None:
        """
        Calculates additional parameters (frequency_undamped, damping_ratio)
        based on values in self.param_optimal_dict.
        """
        try:
            frequency_damped = self.param_optimal_dict["frequency_damped"]
            damping_coeff = self.param_optimal_dict["damping_coeff"]

            freq_undamped = calc_frequency_undamped(frequency_damped, damping_coeff)
            damp_ratio = calc_damping_ratio(damping_coeff, frequency_damped)

            self.param_optimal_dict.update({
                "frequency_undamped": freq_undamped,
                "damping_ratio": damp_ratio
            })

        except KeyError as e:
            logger.warning(f"Missing parameter for derived calculation: {e}")
        except Exception as e:
            logger.warning(f"Failed to calculate derived parameters: {e}", exc_info=True)

    def calc_metrics(self, metrics_warning: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]]) -> Dict[
        str, float]:
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
            Dict[str, float]: A dictionary of calculated metrics.
        """
        try:
            # Generate fitted values using the optimal parameters
            fitted_values = damped_osc(self.df_fit['Sec_Since_Start'], *self.param_optimal)

            # Calculate similarity metrics between the original and fitted data
            self.metrics = SimilarityMetrics.calc(self.df_fit[self.sensor_name], pd.Series(fitted_values))

            # Check if any metrics exceed the provided warning thresholds
            warnings = self.metrics.check_thresholds(metrics_warning)
            for warning in warnings:
                logger.warning(f"{self.measurement.id}_{self.sensor_name}: {warning}")
            self.metric_warning = bool(warnings)

            # Convert metrics to a dictionary and return
            self.metrics_dict = self.metrics.to_dict()
            return self.metrics_dict

        except Exception as e:
            logger.error(f"Error in calculating metrics: {e}", exc_info=True)
            return {}

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

    def plot_fit(self, dir_add: Optional[str] = None, prefix_warning: bool = True,
                 metrics_to_plot: Optional[List[str]] = None) -> None:
        """
        Plots the fitting results and saves the plot.

        Parameters:
        - dir_add: Optional sub-directory name for saving the plot.
        - prefix_warning: Boolean flag to indicate if the filename should be prefixed with "warning_" when a warning exists.
        - metrics_to_plot: Optional list of metric names to be plotted from metrics_dict.

        Available metrics in metrics_dict:
        - pearson_r: Pearson correlation coefficient
        - p_value: p-value of the Pearson correlation
        - r2: R-squared, coefficient of determination
        - mse: Mean Squared Error
        - rmse: Root Mean Squared Error
        - nrmse: Normalized RMSE (Normalized Root Mean Squared Error)
        - cv: Coefficient of Variation
        - mae: Mean Absolute Error
        - nmae: Normalized MAE (Normalized Mean Absolute Error)
        """
        # Default value for metrics_to_plot if not provided
        if metrics_to_plot is None:
            metrics_to_plot = self.CONFIG.Oscillation.metrics_to_plot

        try:
            # Filter the metrics_dict based on metrics_to_plot
            filtered_metrics_dict = {}
            for metric in metrics_to_plot:
                if metric in self.metrics_dict:
                    filtered_metrics_dict[metric] = self.metrics_dict[metric]
                else:
                    logger.warning(f"Metric '{metric}' not found in metrics_dict and will be skipped.")

            # Erstellen des Plots
            fig = plot_oscillation.plot_osc_fit(self.df_fit, self.df, self.sensor_name, self.measurement.id,
                                                self.param_optimal_dict, self.param_optimal,
                                                filtered_metrics_dict, metrics_warning=self.metric_warning,
                                                peaks=self.peaks, valleys=self.valleys)

            # Dateiname entsprechend dem metric_warning Status und prefix_warning Flag
            filename = f"{self.measurement.id}_{self.sensor_name}_{self.measurement.file_name}"
            if self.metric_warning and prefix_warning:
                filename = "warning_" + filename

            # Speichern des Plots
            self.PLOT_MANAGER.save_plot(fig, filename, subdir=f"osc_fit_1{dir_add}")
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
