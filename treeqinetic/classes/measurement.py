from pathlib import Path
import datetime
from typing import List, Dict, Tuple, Union, Optional, Any

import pandas as pd

from .base_class import BaseClass
from .oscillation import Oscillation
from ..plotting import plot_measurement as plt_m
from ..analyse.select_oscillation import calculate_slope, find_oscillation_start, extract_oscillation

from kj_core import get_logger

logger = get_logger(__name__)


class Measurement(BaseClass):
    """
    A class to represent a single Measurement.
    """

    counter = 0  # Class variable to keep track of the number of instances created

    def __init__(self, file_path: Path, project: Dict[str, Any], tree: Dict[str, Any], date: [str, Any],
                 load: [str, Any], test: [str, Any], data: pd.DataFrame):
        """
        Constructs all the necessary attributes for the Measurement object.
        """
        super().__init__()
        Measurement.counter += 1
        self.id: int = Measurement.counter
        self.file_path: Path = Path(file_path)
        self.file_name: str = self.file_path.name

        # Initialize empty attributes
        self.project = project
        self.tree = tree
        self.date = date
        self.load = load
        self.test = test
        self.data = data

        self.datetime_start = None
        self.datetime_end = None
        self.duration = None
        self.length = None

        # Dictionary to store Oscillation objects
        self.oscillations: Dict[str, Oscillation] = {}

    def __str__(self) -> str:
        return f"Measurement: '{self.file_name}', ID: {self.id}, Start '{self.datetime_start}' to '{self.datetime_end}'"

    @classmethod
    def read_txt(cls, file_path) -> Optional['Measurement']:
        """Read data from file and populate the instance attributes."""

        def convert_to_number(s: str) -> Any:
            """Versucht, einen String in eine Ganzzahl oder eine Fließkommazahl zu konvertieren."""
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s.replace(',', '.'))
                except ValueError:
                    return s

        def get_key_value(line: str) -> Optional[Tuple[str, Any]]:
            """Extrahiert Schlüssel-Wert-Paare aus einer Zeile, getrennt durch Tabs."""
            key_value = line.strip().split('\t')

            if not key_value:
                return None

            k = key_value[0]
            if len(key_value) == 2:
                v = convert_to_number(key_value[1])
            elif len(key_value) > 2:
                v = [convert_to_number(element) for element in key_value[1:]]
            else:
                v = None
            return k, v

        project = {}
        tree = {}
        date = {}
        load = {}
        test = {}
        data = pd.DataFrame

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                section = None
                for line in file:
                    # check for section headers
                    if line.startswith('['):
                        section = line.strip()[1:-1]
                    # skip empty lines
                    elif len(line.strip()) == 0:
                        pass
                    # project section
                    elif section == 'project':
                        key, value = get_key_value(line)
                        project[key] = value
                    # tree section
                    elif section == 'tree':
                        key, value = get_key_value(line)
                        tree[key] = value
                    # date section
                    elif section == 'date':
                        key, value = get_key_value(line)
                        if key == 'date_insp':
                            date['date_insp'] = datetime.date(int(value[0]), int(value[1]), int(value[2]))
                        else:
                            date[key] = value
                    # load section
                    elif section == 'load':
                        key, value = get_key_value(line)
                        load[key] = value
                    # test section
                    elif section == 'test':
                        key, value = get_key_value(line)
                        test[key] = value
                    # data section
                    elif section == 'Data':
                        header = line.strip().split('\t')
                        data = pd.read_csv(file, delimiter='\t', decimal=',', names=header)
                        data = data.apply(pd.to_numeric, errors='ignore')

            measurement = cls(file_path, project, tree, date, load, test, data)
            measurement.get_time()
            measurement.data.insert(0, 'ID', measurement.id)

            logger.info(f"Read TXT for {measurement} successful.")
            return measurement

        except Exception as e:
            logger.error(f"Failed to read TXT file: '{file_path}'. Error: {e}")
            return None

    def get_time(self) -> None:
        """Calculate and update the time-related columns in the DataFrame."""
        # Calculate hours, minutes, and seconds from the "Time" column
        hours = self.data['Time'] // 3600
        minutes = (self.data['Time'] % 3600) // 60
        seconds = self.data['Time'] % 60

        # Add new columns "hour", "minute", and "second"
        self.data['Hour'] = hours
        self.data['Minute'] = minutes
        self.data['Second'] = seconds

        # Add a new column "datetime" that contains both the date and time
        self.data.insert(1, 'Datetime',
                         pd.to_datetime(self.data[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']]))
        self.data = self.data.drop(['Time', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second'], axis=1)

        # add start and end time of measurement
        self.datetime_start = min(self.data['Datetime'])
        self.datetime_end = max(self.data['Datetime'])
        self.duration = self.datetime_end - self.datetime_start
        self.length = self.data.__len__()

        self.data.insert(2, 'Sec_Since_Start',
                         (self.data['Datetime'] - self.datetime_start).dt.total_seconds().astype(float))

    def select_oscillations(self, sensor_names: List[str], min_time_default: float = 60, min_value: float = 50,
                            threshold_slope: float = -25, duration: float = 20):
        """Identify and store oscillation data for the given sensor names."""
        df = self.data

        # Entfernen von NaN-Werten, sortieren nach der Zeitspalte und indexreset
        df.dropna(subset=sensor_names, inplace=True)
        df.sort_values(by=['Sec_Since_Start'], inplace=True)
        df.reset_index(drop=True, inplace=True)

        for sensor_name in sensor_names:

            # Calculate slope
            calculate_slope(df, sensor_name)

            # Find the start index of the oscillation period
            start_index = find_oscillation_start(df, sensor_name, threshold_slope, min_time_default, min_value)

            # Extract the oscillation period
            if start_index is not None:
                oscillation_df = extract_oscillation(df, sensor_name, start_index, duration)
            else:
                oscillation_df = None

            # Store the data in an Oscillation object
            self.oscillations[sensor_name] = Oscillation(self, sensor_name, start_index, oscillation_df)

    def plot_multi_sensors(self, sensor_names: List[str], time_start: Union[None, datetime.datetime] = None,
                           time_end: Union[None, datetime.datetime] = None) -> None:
        """Plot data for multiple sensors between given start and end times."""
        try:
            plt_m.plot_multi_sensors(self.PLOT_MANAGER, self, sensor_names=sensor_names, time_start=time_start, time_end=time_end)
            logger.info(f"plot_multi_sensors for measurement: '{self}' successful.")
        except Exception as e:
            logger.error(f"Failed to plot_multi_sensors: '{self}'. Error: {e}")

    def plot_select_oscillation_single(self, sensor_names: List[str]) -> None:
        """Plot selected oscillation data for a single sensor."""

        for sensor_name in sensor_names:
            if sensor_name in self.oscillations:
                oscillation = self.oscillations[sensor_name]
                try:
                    plt_m.plot_select_oscillation_single(self.PLOT_MANAGER, self, oscillation)
                    logger.info(f"plot_select_oscillation for measurement: '{self}' for {sensor_name} successful.")
                except Exception as e:
                    logger.error(f"Failed to plot_select_oscillation: '{self}' for {sensor_name}. Error: {e}")
            else:
                logger.warning(f"No oscillation data found for sensor: {sensor_name}")

    def plot_select_oscillation_multi(self, sensor_names: List[str]) -> None:
        """Plot selected oscillation data for multiple sensors."""
        oscillations_to_plot = []
        for sensor_name in sensor_names:
            if sensor_name in self.oscillations:
                oscillations_to_plot.append(self.oscillations[sensor_name])
        try:
            plt_m.plot_select_oscillation_multi(self.PLOT_MANAGER, self, oscillations_to_plot)
            logger.info(f"Multi-sensor oscillation plot for measurement: '{self}' successful.")
        except Exception as e:
            logger.error(f"Failed to create multi-sensor oscillation plot: '{self}'. Error: {e}")
