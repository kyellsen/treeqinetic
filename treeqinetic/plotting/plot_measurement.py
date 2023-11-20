import matplotlib.pyplot as plt


def plot_multi_sensors(plot_manager, measurement, sensor_names: list, time_start=None, time_end=None):
    if time_start is None:
        time_start = 0
    if time_end is None:
        time_end = measurement.length

    file_name = str(measurement.file_name).replace(".", "_")

    data = measurement.data[
        (measurement.data['Sec_Since_Start'] > time_start) & (measurement.data['Sec_Since_Start'] < time_end)]

    # Erstelle die Plotfigur und die Achsen
    fig, ax = plt.subplots()
    for sensor_name in sensor_names:
        ax.plot(data['Sec_Since_Start'], data[sensor_name], label=sensor_name)
        ax.scatter(data['Sec_Since_Start'], data[sensor_name], s=0.2, c='black', zorder=2)
    plt.xlabel('Sec_Since_Start')
    plt.ylabel('Dehnung / Neigung [µm/°]')
    plt.title(f'Plot of {file_name} vs. time')
    plt.legend()
    plt.tight_layout()
    plot_manager.save_plot(fig, f"plot_sensors_{file_name}_{measurement.id}", subdir="multi_sensors_vs_time_1")
    plt.close()


# Hilfsfunktion zur Konfiguration der Plots
def configure_plot(ax, x_data, y_data, sensor_name, x_label, y_label, title, legend_label, color='b'):
    ax.plot(x_data, y_data, label=legend_label, color=color)
    ax.scatter(x_data, y_data, s=5, c='black', zorder=2)
    ax.set_title(title.format(sensor_name))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True)


# Einzelner Sensor
def plot_select_oscillation_single(plot_manager, measurement, oscillation):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    df = measurement.data
    sensor_name = oscillation.sensor_name
    configure_plot(ax1, df['Sec_Since_Start'], df[sensor_name], sensor_name, 'Time (Seconds)', sensor_name,
                   'Original Data ({})', 'Original Data')

    if oscillation.df_orig is not None:
        configure_plot(ax2, oscillation.df_orig['Sec_Since_Start'], oscillation.df_orig[sensor_name], sensor_name,
                       'Time (Seconds)', sensor_name, 'Isolated Oscillation ({})', 'Isolated Oscillation')

    # Skaliere die Y-Achsen gleich
    min_y = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
    max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
    ax1.set_ylim(min_y, max_y)
    ax2.set_ylim(min_y, max_y)

    plt.tight_layout()
    plot_manager.save_plot(fig, f"{measurement.file_name}_{measurement.id}_{sensor_name}",
                           subdir="select_oscillation_single_1")
    plt.close(fig)


# Mehrere Sensoren
def plot_select_oscillation_multi(plot_manager, measurement, oscillations):
    fig, axs = plt.subplots(len(oscillations), 2, figsize=(15, 6 * len(oscillations)))

    for i, oscillation in enumerate(oscillations):
        df = measurement.data
        sensor_name = oscillation.sensor_name

        # Achsenobjekte für den aktuellen Sensor
        ax1 = axs[i, 0]
        ax2 = axs[i, 1]

        # Konfiguriere die Plots
        configure_plot(ax1, df['Sec_Since_Start'], df[sensor_name], sensor_name, 'Time (Seconds)', sensor_name,
                       'Original Data ({})', 'Original Data')
        if oscillation.df_orig is not None:
            configure_plot(ax2, oscillation.df_orig['Sec_Since_Start'], oscillation.df_orig[sensor_name], sensor_name,
                           'Time (Seconds)', sensor_name, 'Isolated Oscillation ({})', 'Isolated Oscillation')

        # Skaliere die Y-Achsen gleich
        min_y = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        max_y = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(min_y, max_y)
        ax2.set_ylim(min_y, max_y)

    plt.tight_layout()
    plot_manager.save_plot(fig, f"{measurement.file_name}_{measurement.id}_multi_sensor",
                           subdir="select_oscillation_multi_1")
    plt.close(fig)
