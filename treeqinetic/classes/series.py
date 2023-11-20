from pathlib import Path
import pandas as pd

from .base_class import BaseClass
from .measurement import Measurement

from kj_core import get_logger

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

    def plot_single_oscillations_for_measurements(self, sensor_names: list):
        for measurement in self.measurements:
            logger.info(f"Plot single Oscillations for {measurement}")
            measurement.plot_select_oscillation_single(sensor_names)

    def plot_multi_oscillations_for_measurements(self, sensor_names: list):
        for measurement in self.measurements:
            logger.info(f"Plot multiple Oscillations for {measurement}")
            measurement.plot_select_oscillation_multi(sensor_names)

    def get_oscillations_list(self):
        oscillation_list = []
        for measurement in self.measurements:
            oscillation_list.extend(measurement.oscillations.values())
        return oscillation_list

    def get_oscillations_df(self):
        oscillations = self.get_oscillations_list()
        print(oscillations)
        data = []

        for osc in oscillations:
            row = {
                'id': osc.measurement.id,
                'file_name': osc.measurement.file_name,
                'sensor_name': osc.sensor_name,
                'sample_rate': osc.sample_rate,
                'offset': osc.offset,
                'min': osc.min,
                'max': osc.max,
                'amplitude': osc.amplitude,
                'amplitude_2': osc.amplitude_2,
                'frequency': osc.frequency,
                'damping_coeff_avg': osc.damping_coeff_avg,
                'damping_coeff_peaks': osc.damping_coeff_peaks,
                'damping_coeff_valleys': osc.damping_coeff_valleys,
            }
            data.append(row)
        df = pd.DataFrame(data)

        return df
