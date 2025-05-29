import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_integral(
    df: pd.DataFrame,
    sensor_name: str,
    measurement_id: int,
    results: dict
):
    """
    Plots raw and centered sensor data along with colored areas representing
    positive and negative integrals. Annotates the plot with computed values.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'Sec_Since_Start', sensor_name, and 'value_centered'.
    - sensor_name (str): Column name of the sensor data in df.
    - results (dict): Dictionary returned by build_integral with keys:
        'pos_int', 'neg_int', 'abs_int', 'ratio', 'unit'.

    Returns:
    - Figure
    """
    cp = sns.color_palette('bright')
    # Unpack results
    pos_int = results['integral_positiv']
    neg_int = results['integral_negativ']
    abs_int = results['integral_abs']
    ratio = results['integral_ratio']
    unit = results.get('integral_unit', '')

    times = df['Sec_Since_Start']
    raw = df[sensor_name]
    cent = df['cent']

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times, raw,  label='Raw data',    linewidth=1, c="grey")
    ax.plot(times, cent, label='Centered',     linewidth=1, c="black")

    # Fill-positive and negative areas
    ax.fill_between(
        times, cent, 0,
        where=cent >= 0, interpolate=True,
        alpha=0.4,
        color='#1ac938',
        label=f"Positive: {pos_int:.2f} {unit}",
    )
    ax.fill_between(
        times, cent, 0,
        where=cent <= 0, interpolate=True,
        alpha=0.4,
        color='#e8000b',
        label=f"Negative: {neg_int:.2f} {unit}"
    )

    plt.title(f"Berechnung der Integrale {measurement_id} - {sensor_name}", fontsize=14)
    plt.xlabel("Zeit $t$ [s]")
    plt.ylabel(f"Absolute Randfaserdehnung $\\Delta$L [$\\mu$m] / Neigung $\\varphi$ [°]")
    ax.grid(True, linestyle='--', alpha=0.5)

    # Annotation box
    text = (
        f"Total |Integral|: {abs_int:.2f} {unit}\n"
        f"Pos/Neg ratio: {ratio:.2f}"
    )
    ax.text(
        0.98, 0.95, text,
        ha='right', va='top', transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', alpha=0.3, facecolor='white', edgecolor="black")
    )

    ax.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    return fig
