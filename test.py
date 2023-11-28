import treeqinetic as ptq
from pathlib import Path

# Main
main_path = Path(r"C:\kyellsen\005_Projekte\2023_Kronensicherung_Plesse")
analyse_name = r"2023_Kronensicherung_Plesse_2023-11-14"
data_path = main_path / "020_Daten"  # Für alle Daten-Importe des Projektes gemeinsam
working_directory = main_path / "030_Analysen" / analyse_name / "working_directory"  # Für alle Daten-Exporte des Projektes gemeinsam

ptq_working_directory = working_directory / 'PTQ'
ptq.setup(working_directory=ptq_working_directory, log_level="INFO")

ptq_data_path = data_path / 'PTQ/data_txt'
ptq_series = ptq.classes.Series(name=analyse_name, path=ptq_data_path)

elasto_names = ["Elasto(95)", "Elasto(98)", "Elasto(92)", "Elasto(90)"]

# ptq_series.plot_measurement_sensors(sensor_names=elasto_names)

ptq_series.get_oscillations(sensor_names=elasto_names)

# ptq_series.plot_oscillations_for_measurements(sensor_names=elasto_names, combined=True)
# ptq_series.plot_oscillations_for_measurements(sensor_names=elasto_names, combined=False)

# Definiert eine Liste von Parameternamen
param_labels = ["initial_amplitude", "damping_coeff", "angular_frequency", "phase_angle", "y_shift"]

# Definiert die Anfangswerte für jeden Parameter
initial_values = [180, 0.35, 0.44, 0, 0]

# Definiert die Grenzwerte für jeden Parameter
bounds_values = [(125, 400), (0.2, 5), (0.25, 0.55), (-0.2, 0.2), (-30, 30)]

# Erstellt das initial_params Dictionary mit einer Dictionary-Comprehension
initial_params = {label: value for label, value in zip(param_labels, initial_values)}

# Erstellt das param_bounds Dictionary auf ähnliche Weise
param_bounds = {label: bounds for label, bounds in zip(param_labels, bounds_values)}

# Angepasstes metrics_warning-Dictionary
metrics_warning = {
    "mse": (0, 1000),  # Keine Untergrenze, Obergrenze bei 1000
    "mae": (0, 50),    # Keine Untergrenze, Obergrenze bei 50
    "rmse": (0, 100),  # Keine Untergrenze, Obergrenze bei 100
    "r2": (0.6, 1)     # Untergrenze bei 0.6, keine Obergrenze
}


osc_list = ptq_series.get_oscillations_list()


# Dictionary zum Sammeln der Metriken
collected_metrics = {"mse": [], "mae": [], "rmse": [], "r2": []}

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# # Durchführen der Durchläufe
# for osc in osc_list:
#     metrics_dict = osc.fit_and_plot(initial_params, param_bounds, metrics_warning)
#     if metrics_dict:
#         for metric in collected_metrics:
#             collected_metrics[metric].append(metrics_dict.get(metric, np.nan))
#
#
# # Statistische Analyse der gesammelten Metriken
# analyzed_metrics = {metric: {"mean": np.mean(values), "median": np.median(values),
#                              "std_dev": np.std(values)} for metric, values in collected_metrics.items()}
#
# # Jetzt enthält analyzed_metrics statistische Kennzahlen für jede Metrik
# print(analyzed_metrics)
#
#
# # Konvertierung der Daten in ein pandas DataFrame
# metrics_df = pd.DataFrame(analyzed_metrics).T  # Transponieren, um Metriken als Zeilen zu haben
#
# # Erstellen des Plots für die Durchschnittswerte
# plt.figure(figsize=(10, 6))
# sns.barplot(data=metrics_df, x=metrics_df.index, y='mean')
# plt.title('Mean')
# plt.ylabel('Mean')
# plt.xlabel('Metrik')
# plt.show()
#
# # Erstellen des Plots für die Durchschnittswerte
# plt.figure(figsize=(10, 6))
# sns.barplot(data=metrics_df, x=metrics_df.index, y='median')
# plt.title('Median')
# plt.ylabel('Median')
# plt.xlabel('Metrik')
# plt.show()
#
#
# # Erstellen des Plots für die Standardabweichungen
# plt.figure(figsize=(10, 6))
# sns.barplot(data=metrics_df, x=metrics_df.index, y='std_dev')
# plt.title('StdDev')
# plt.ylabel('StdDev')
# plt.xlabel('Metrik')
# plt.show()

