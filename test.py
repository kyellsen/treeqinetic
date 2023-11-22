import treeqinetic as ptq
from pathlib import Path


# Main
main_path = Path(r"C:\kyellsen\005_Projekte\2023_Kronensicherung_Plesse")
analyse_name = r"2023_Kronensicherung_Plesse_2023-11-14"
data_path = main_path / "020_Daten" # Für alle Daten-Importe des Projektes gemeinsam
working_directory = main_path / "030_Analysen" / analyse_name / "working_directory" # Für alle Daten-Exporte des Projektes gemeinsam


ptq_working_directory = working_directory / 'PTQ'
ptq.setup(working_directory=ptq_working_directory, log_level="INFO")

ptq_data_path = data_path  / 'PTQ/data_txt'
ptq_series = ptq.classes.Series(name=analyse_name, path=ptq_data_path)

elasto_names = ["Elasto(95)", "Elasto(98)", "Elasto(92)", "Elasto(90)"]


ptq_series.plot_measurement_sensors(sensor_names=elasto_names)


ptq_series.get_oscillations(sensor_names=elasto_names)


ptq_series.plot_single_oscillations_for_measurements(sensor_names=elasto_names) # stell jeden sensor einzeln dar
