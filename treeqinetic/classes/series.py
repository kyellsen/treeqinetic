from pathlib import Path
import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import json

from .base_class import BaseClass
from .measurement import Measurement
from .oscillation import Oscillation

from ..plotting.plot_fit_error import plot_error_histogram, plot_error_qq, plot_error_violin

from kj_logger import get_logger

logger = get_logger(__name__)


class Series(BaseClass):
    """
    Repräsentiert eine Serie von Messungen aus Textdateien in einem Verzeichnis.

    Attributes:
        name (str): Name der Serie.
        path (Path): Pfad zum Verzeichnis mit den Messungsdateien.
        measurement_files_paths (List[Path]): Liste aller .txt-Dateipfade im Verzeichnis.
        measurement_files (List[str]): Liste aller Dateinamen der .txt-Dateien.
        measurements (List[Measurement]): Liste der eingelesenen Measurement-Instanzen.
        measurements_count (int): Anzahl der eingelesenen Messungen.
        df_list (List[pd.DataFrame]): Liste der DataFrames aller Messungen.
        df (pd.DataFrame): Zusammengeführtes DataFrame aller Messungen.
        osc_df (Optional[pd.DataFrame]): DataFrame der Oszillationsdaten (initial None).
        osc_errors_dict_all (Optional[Dict]): Fehlerdaten über alle Sensoren (initial None).
        osc_errors_dict_by_sensor (Optional[Dict]): Fehlerdaten gruppiert nach Sensor (initial None).
    """

    def __init__(self, name: str, path: str) -> None:
        """
        Initialisiert eine neue Series-Instanz.

        Args:
            name (str): Name der Serie.
            path (str): Pfad zum Verzeichnis, das die .txt-Messungsdateien enthält.

        Raises:
            ValueError: Wenn der angegebene Pfad nicht existiert oder kein Verzeichnis ist.
        """
        super().__init__()
        self.name: str = name

        self.path: Path = Path(path)
        if not self.path.exists():
            logger.error(f"Der Pfad '{self.path}' existiert nicht.")
            raise ValueError(f"Der Pfad '{self.path}' existiert nicht.")
        if not self.path.is_dir():
            logger.error(f"Der Pfad '{self.path}' ist kein Verzeichnis.")
            raise ValueError(f"Der Pfad '{self.path}' ist kein Verzeichnis.")

        # Sammle alle .txt-Dateien im Verzeichnis
        self.measurement_files_paths: List[Path] = [
            f for f in self.path.iterdir() if f.is_file() and f.suffix.lower() == '.txt'
        ]
        self.measurement_files: List[str] = [f.name for f in self.measurement_files_paths]

        if not self.measurement_files_paths:
            logger.warning(f"Keine '.txt'-Dateien im Verzeichnis '{self.path}' gefunden.")

        self.measurements: List[Measurement] = []

        for measurement_file_path in self.measurement_files_paths:
            try:
                # Erstelle eine Measurement-Instanz aus der Datei
                measurement = Measurement.read_txt(file_path=measurement_file_path)
                self.measurements.append(measurement)
            except Exception as e:
                logger.error(f"Fehler beim Einlesen der Datei '{measurement_file_path}': {e}")

        self.measurements_count: int = len(self.measurements)

        # Erstelle Liste von DataFrames und kombiniertes DataFrame
        self.df_list: List[pd.DataFrame] = self.get_measurements_df_list()
        try:
            self.df: pd.DataFrame = self.get_measurements_df()
        except ValueError:
            # Falls df_list leer ist, lege ein leeres DataFrame an
            self.df = pd.DataFrame()
            logger.warning("Keine Daten zum Zusammenführen gefunden; 'df' ist leer.")

        # Oszillationsdaten und Fehlerinitialisierung
        self.osc_df: Optional[pd.DataFrame] = None
        self.osc_errors_dict_all: Optional[Dict] = None
        self.osc_errors_dict_by_sensor: Optional[Dict] = None

    def __str__(self) -> str:
        """
        Liefert eine lesbare Darstellung der Series-Instanz.

        Returns:
            str: Beschreibung mit Name, Anzahl und Dateinamen der Messungen.
        """
        return f"Series: '{self.name}' mit {self.measurements_count} Messungen: {self.measurement_files}"

    def get_measurements_df_list(self) -> List[pd.DataFrame]:
        """
        Ermittelt für jede Measurement-Instanz deren DataFrame.

        Returns:
            List[pd.DataFrame]: Liste aller DataFrames der Measurements.
        """
        df_list: List[pd.DataFrame] = []
        for measurement in self.measurements:
            if hasattr(measurement, 'data') and isinstance(measurement.data, pd.DataFrame):
                df_list.append(measurement.data)
            else:
                logger.warning(f"Measurement '{measurement}' hat kein valides DataFrame-Attribut 'data'.")
        return df_list

    def get_measurements_df(self) -> pd.DataFrame:
        """
        Verkettet alle DataFrames aus get_measurements_df_list zu einem einzigen DataFrame.

        Returns:
            pd.DataFrame: Zusammengeführtes DataFrame aller Messdaten.

        Raises:
            ValueError: Wenn keine DataFrames zum Zusammenführen vorhanden sind.
        """
        if not self.df_list:
            raise ValueError("Keine DataFrames vorhanden, um sie zusammenzuführen.")
        try:
            combined_df = pd.concat(self.df_list, ignore_index=True)
        except Exception as e:
            logger.error(f"Fehler beim Zusammenführen der DataFrames: {e}")
            raise
        return combined_df

    def plot_measurement_sensors(
        self,
        sensor_names: List[str],
        time_start: Optional[float] = None,
        time_end: Optional[float] = None
    ) -> None:
        """
        Plottet den Zeitverlauf der angegebenen Sensoren für jede Measurement-Instanz.

        Args:
            sensor_names (List[str]): Liste der Sensor-Namen, die geplottet werden sollen.
            time_start (Optional[float]): Startzeitpunkt für den Plot (Standard: None).
            time_end (Optional[float]): Endzeitpunkt für den Plot (Standard: None).

        Raises:
            ValueError: Wenn sensor_names leer ist oder kein List-Typ.
        """
        if not isinstance(sensor_names, list) or not sensor_names:
            logger.error("Die Liste 'sensor_names' darf nicht leer sein und muss vom Typ List[str] sein.")
            raise ValueError("sensor_names muss eine nicht-leere Liste von Zeichenketten sein.")

        for measurement in self.measurements:
            try:
                measurement.plot_multi_sensors(sensor_names, time_start, time_end)
            except Exception as e:
                logger.error(f"Fehler beim Plotten für Measurement '{measurement}': {e}")


    def get_oscillations(self, sensor_names: List[str], **kwargs):
        """
        Meta-function to select oscillation data for multiple measurements.

        This function iterates over all measurements and applies the select_oscillations method to each instance of measurement.

        Parameters:
        ----------
        sensor_names : List[str]
            A list of sensor names for which the oscillation data needs to be identified.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the select_oscillations function. These can include:

            - min_time_default: float
                The minimum time period to be considered for identifying oscillations.
            - min_value: float
                The minimum value threshold for sensor data to be considered valid.
            - threshold_slope: float
                The slope threshold to determine the start of an oscillation.
            - duration: float
                The duration for which the oscillation data is to be extracted.

        Returns:
        -------
        None
        """
        for measurement in self.measurements:
            logger.info(f"\n Select Oscillations for {measurement}")
            measurement.select_oscillations(sensor_names, **kwargs)

    def plot_oscillations_for_measurements(self, sensor_names: list, combined: bool = False):
        for measurement in self.measurements:
            logger.info(f"Plot Oscillations for {measurement}")
            measurement.plot_select_oscillations(sensor_names, combined)

    def get_oscillations_list(self) -> List[Oscillation]:
        """
        Returns a flat list of all Oscillation objects from all measurements.
        """
        oscillation_list = []
        try:
            for measurement in self.measurements:
                # measurement.oscillations.values() sind alle Oscillation-Objekte
                # der jeweiligen Messung
                oscillation_list.extend(measurement.oscillations.values())
        except Exception as e:
            logger.error(f"Failed to gather oscillations list: {e}")
        return oscillation_list


    def _collect_oscillation_data(self, osc: Oscillation) -> Dict:
        """
        Helper function to collect measurement-related data from a single Oscillation object.
        Includes sensor extremes, param_optimal_dict, metrics_dict etc.

        Returns:
            Dict mit den wichtigsten Informationen zur Schwingung.
        """
        data_row = {}
        try:
            sensor_name = osc.sensor_name

            # Basiswerte
            data_row = {
                'id': osc.measurement.id,
                'file_name': osc.measurement.file_name,
                'sensor_name': sensor_name,
                'sample_rate': osc.sample_rate,
                'max_strain': osc.measurement.data[sensor_name].max(),
                'max_compression': osc.measurement.data[sensor_name].min(),
                'max_strain_osc': osc.max_value,
                'max_compression_osc': osc.min_value,
                'm_amplitude': osc.m_amplitude,
                'm_amplitude_2': osc.m_amplitude_2,
                'metrics_warning': osc.metric_warning,
            }

            # Werte aus param_optimal_dict hinzufügen
            if osc.param_optimal_dict:
                data_row.update(osc.param_optimal_dict)

            # Werte aus metrics_dict hinzufügen
            if osc.metrics_dict:
                data_row.update(osc.metrics_dict)

            # Werte aus integral_dict hinzufügen
            if osc.integral_dict:
                data_row.update(osc.integral_dict)

        except Exception as e:
            logger.error(f"Error while collecting osc data for {osc}: {e}")

        return data_row

    def _collect_oscillation_optimization_details(self, osc: Oscillation) -> Dict:
        """
        Helper function to collect optimization details (e.g., nit, nfev) from a single Oscillation object.
        Includes optional metrics_dict if present.

        Returns:
            Dict mit grundlegenden Angaben plus optimization_details.
        """
        data_row = {}
        try:
            # Nur wenn optimization_details existiert
            if osc.optimization_details is not None:
                data_row = {
                    'id': osc.measurement.id,
                    'file_name': osc.measurement.file_name,
                    'sensor_name': osc.sensor_name,
                }
                # Optimierungsdetails hinzufügen
                data_row.update(osc.optimization_details)

                # metrics_dict ggf. hinzufügen
                if osc.metrics_dict:
                    data_row.update(osc.metrics_dict)
        except Exception as e:
            logger.error(f"Error while collecting optimization details for {osc}: {e}")

        return data_row

    def get_oscillations_df(self) -> pd.DataFrame:
        """
        Creates a DataFrame containing measurement and fit information for all Oscillation objects.

        Returns:
            pd.DataFrame: DataFrame mit Spalten wie ID, file_name, sensor_name,
                          sowie gemessenen und berechneten Werten (z. B. max_strain, param_optimal_dict usw.).
        """
        oscillations = self.get_oscillations_list()
        data_rows = []

        for osc in oscillations:
            row = self._collect_oscillation_data(osc)
            if row:  # Nur wenn nicht leer
                data_rows.append(row)

        df = pd.DataFrame(data_rows)
        # Datentyp anpassen (optional)
        if not df.empty and 'sensor_name' in df.columns:
            df['sensor_name'] = df['sensor_name'].astype('category')

        return df

    def get_osc_optimization_details_df(self) -> pd.DataFrame:
        """
        Creates a DataFrame from the Optimization-Details (self.optimization_details)
        of all Oscillation objects. Includes optional metrics_dict if present.

        Returns:
            pd.DataFrame: Columns for ID, file_name, sensor_name,
                          plus the fields from osc.optimization_details and metrics_dict.
        """
        oscillations = self.get_oscillations_list()
        data_rows = []

        for osc in oscillations:
            row = self._collect_oscillation_optimization_details(osc)
            if row:  # Nur wenn nicht leer
                data_rows.append(row)

        df = pd.DataFrame(data_rows)
        return df

    @staticmethod
    def create_oscillations_data_dict() -> Dict[str, dict]:
        """
        Loads and returns the PTQ scillations data dictionary from JSON.

        Returns
        -------
        Dict[str, dict]
            Data dictionary mapping column names to their metadata description.
        """
        try:
            json_path = Path(__file__).parent.parent / "ptq_oscillations_data_dict.json"
            if not json_path.exists():
                logger.warning(f"Data dictionary not found at: {json_path}")
                return {}

            with open(json_path, "r", encoding="utf-8") as f:
                data_dict = json.load(f)

            logger.info(f"Data dictionary loaded with {len(data_dict)} entries.")
            return data_dict

        except Exception as e:
            logger.error(f"Error loading data dictionary: {e}")
            return {}

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
