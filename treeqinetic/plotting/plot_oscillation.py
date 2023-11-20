import numpy as np
import matplotlib.pyplot as plt


from ..analyse.fitting_functions import exp_decreasing, damped_oscillation


def plot_basic_data(ax, oscillation):
    ax.plot(oscillation.df['Sec_Since_Start'], oscillation.df[oscillation.sensor_name], label='Data', alpha=0.7)
    ax.scatter(oscillation.df['Sec_Since_Start'], oscillation.df[oscillation.sensor_name], s=10, c='black', zorder=2,
               alpha=0.7)


def extract_peak_valley_info(oscillation):
    return (
        np.array([peak['time'] for peak in oscillation.peaks]),
        np.array([peak['value'] for peak in oscillation.peaks]),
        np.array([valley['time'] for valley in oscillation.valleys]),
        np.array([valley['value'] for valley in oscillation.valleys])
    )

def plot_peak_valley_points(ax, peak_times, peak_values, valley_times, valley_values):
    ax.scatter(peak_times, peak_values, c='red', marker='v', zorder=3, label='Peaks')
    ax.scatter(valley_times, valley_values, c='green', marker='^', zorder=3, label='Valleys')
    ax.scatter(peak_times[1], peak_values[1], c='orange', marker='v', zorder=5, label='Peak1')


def plot_amplitude_error_bars(ax, oscillation, peak_times, peak_values):
    if oscillation.amplitude:
        ax.errorbar(peak_times[0], (oscillation.max + oscillation.min) / 2, yerr=oscillation.amplitude, fmt='none',
                    ecolor='purple', capsize=10, label='Amplitude', zorder=4)
    if oscillation.amplitude_2:
        ax.errorbar(peak_times[1], (peak_values[1] + oscillation.min) / 2, yerr=oscillation.amplitude_2, fmt='none',
                    ecolor='green', capsize=10, label='Amplitude_2', zorder=4)

def plot_oscillation(plot_manager, oscillation):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_basic_data(ax, oscillation)
    peak_times, peak_values, valley_times, valley_values = extract_peak_valley_info(oscillation)
    plot_peak_valley_points(ax, peak_times, peak_values, valley_times, valley_values)
    plot_amplitude_error_bars(ax, oscillation, peak_times, peak_values)
    plt.title(f"Time vs {oscillation.sensor_name}")
    plt.xlabel("Time (Sec_Since_Start)")
    plt.ylabel(f"Measurement ({oscillation.sensor_name})")
    plt.legend()
    plt.tight_layout()
    plot_manager.save_plot(fig,
                           f"{oscillation.measurement.file_name}_{oscillation.measurement.id}_{oscillation.sensor_name}",
                           subdir="oscillation_1")
    plt.close(fig)


def plot_oscillation_with_damping(plot_manager, oscillation):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_basic_data(ax, oscillation)
    peak_times, peak_values, valley_times, valley_values = extract_peak_valley_info(oscillation)

    # Plot damping functions for peaks and valleys, if available
    if hasattr(oscillation, 'damping_coeff_peaks') and hasattr(oscillation, 'damping_coeff_valleys'):
        # Define the time range for plotting damping functions
        min_time = min(min(peak_times), min(valley_times))
        max_time = max(max(peak_times), max(valley_times))
        time_values = np.linspace(min_time, max_time, 500)  # 500 points for smoothness

        ax.plot(time_values, exp_decreasing(time_values - min_time, peak_values[0], oscillation.damping_coeff_peaks),
                label='Damping (Peaks)', linestyle='--', alpha=0.7)
        ax.plot(time_values, exp_decreasing(time_values - min_time, valley_values[0], oscillation.damping_coeff_valleys),
                label='Damping (Valleys)', linestyle='--', alpha=0.7)

    plot_peak_valley_points(ax, peak_times, peak_values, valley_times, valley_values)
    plot_amplitude_error_bars(ax, oscillation, peak_times, peak_values)
    plt.title(f"Time vs {oscillation.sensor_name}")
    plt.xlabel("Time (Sec_Since_Start)")
    plt.ylabel(f"Measurement ({oscillation.sensor_name})")
    plt.legend()
    plt.tight_layout()
    plot_manager.save_plot(fig,
                           f"{oscillation.measurement.file_name}_{oscillation.measurement.id}_{oscillation.sensor_name}",
                           subdir="oscillation_damping_1")
    plt.close(fig)

# # Methode zum Plotten der Messdaten und des Fits
# def plot_oscillation_fitting(oscillation):
#     fig, ax = plt.subplots(figsize=(10, 6))
#     plot_basic_data(ax, oscillation)
#     plt.plot(oscillation.df['Sec_Since_Start'], oscillation.fit_data, label='Fit', linestyle='--')
#     plt.title(f'Vergleich der Messdaten und Fit-Daten ({oscillation.sensor_name})')
#     plt.xlabel('Zeit (s)')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.show()
