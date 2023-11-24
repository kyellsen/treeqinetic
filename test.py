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



param_bounds = {
    'initial_amplitude': (125, 400),  # Min und Max für initial_amplitude
    'damping_coeff': (0.2, 5),  # Min und Max für damping_coeff
    'angular_frequency': (0.25, 0.55),  # Min und Max für angular_frequency
    'phase_angle': (-0.2, 0.2)  # Min und Max für phase_angle
}

initial_params = {
    'initial_amplitude': 180,
    'damping_coeff': 0.35,
    'angular_frequency': 0.44,
    'phase_angle': 0
}

osc_list = ptq_series.get_oscillations_list()


for osc in osc_list:
    osc.fit_and_plot(initial_params, param_bounds)

