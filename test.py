from pathlib import Path
import numpy as np
import pandas as pd
import sys
from typing import List, Tuple
from kj_logger import get_logger, LOG_MANAGER

import treeqinetic as ptq
from treeqinetic import Series

if __name__ == "__main__":
    main_path = Path(r"C:\kyellsen\005_Projekte\2023_Kronensicherung_Plesse")
    analyse_name = r"2023_Kronensicherung_Plesse_2024-05-29"
    data_path = main_path / "020_Daten"  # Für alle Daten-Importe des Projektes gemeinsam
    working_directory = main_path / "030_Analysen" / analyse_name / "working_directory"  # Für alle Daten-Exporte des Projektes gemeinsam

    CONFIG, LOG_MANAGER, PLOT_MANAGER = ptq.setup(working_directory=str(working_directory), log_level="DEBUG")

    ptq_data_path = data_path / 'PTQ/data_txt'
    ptq_series = ptq.classes.Series(name=analyse_name, path=str(ptq_data_path))

    elasto_names = ["Elasto(95)", "Elasto(98)", "Elasto(92)", "Elasto(90)"]

    ptq_series.get_oscillations(sensor_names=elasto_names)

    ptq_oscillations_ls = ptq_series.get_oscillations_list()

    for osc in ptq_oscillations_ls:
        osc.fit(plot=False, plot_error=False, dir_add="_df", clean_peaks=False, interpolate=True)

    all_normalized_errors = ptq_series.plot_osc_errors(plot_qq=True, plot_violin=True, plot_hist=True,
                                                       hist_trim_percent=2)

# param_labels = ptq.CONFIG.Oscillation.param_labels
# metrics_labels = ptq.CONFIG.Oscillation.metrics_labels
# columns_to_plot = param_labels + metrics_labels
#
# # Erstellen Sie für jede Spalte einen eigenen Boxplot
# for column in columns_to_plot:
#     # Erstellen eines neuen Plots
#     plt.figure(figsize=(10, 6))
#
#     # Kombinieren von df und df2 für den aktuellen Parameter in einem DataFrame
#     combined_df = pd.concat([df[column], df2[column]], axis=1)
#     combined_df.columns = ['df', 'df2']
#
#     # Erstellen des Boxplots mit Seaborn
#     sns.boxplot(data=combined_df)
#
#     # Titel und Achsenbeschriftungen hinzufügen
#     plt.title(f'Boxplot für {column}')
#     plt.xlabel('DataFrame')
#     plt.ylabel(column)
#
#     # Anzeigen des Plots
#     plt.show()
