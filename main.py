import treeqinetic as ptq
from treeqinetic.classes import Series

ptq.setup(working_directory=r"C:\kyellsen\006_Packages\test_working_directory_user_ptq", log_level="INFO")

project_name = r"Plesse_Kronensicherung_2023"
ptq_data_path = r"C:\kyellsen\005_Projekte\2023_Kronensicherung_Plesse\020_Daten\PTQ\data_txt"

ptq_series = Series(name=project_name, path=ptq_data_path)

sensor_names = ["Elasto(95)", "Elasto(98)", "Elasto(92)", "Elasto(90)"]

ptq_series.get_oscillations(sensor_names)
# ptq_series.plot_single_oscillations_for_measurements(sensor_names)
# ptq_series.plot_multi_oscillations_for_measurements(sensor_names)

# df = ptq_series.get_oscillation_df()
# df.to_csv('oscillation.csv', index=False)
