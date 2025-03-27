import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Ergänze rechnerische Schwingungsparameter die Rust(2013) genutzt hat


def calc_frequency_undamped(frequency_damped: float, damping_coeff: float) -> Optional[float]:
    """
    Berechnet die ungedämpfte Frequenz auf Basis der gedämpften Frequenz und der Dämpfungskonstante.

    Formel: f₀ = √(f_d² + (d / 2π)²)

    Args:
        frequency_damped (float): Gedämpfte Frequenz [Hz]
        damping_coeff (float): Dämpfungskonstante

    Returns:
        Optional[float]: Ungedämpfte Frequenz [Hz] oder None bei Fehler
    """
    try:
        return np.sqrt(frequency_damped**2 + (damping_coeff / (2 * np.pi))**2)
    except Exception as e:
        logger.warning(f"Could not calculate frequency_undamped: {e}")
        return None


def calc_damping_ratio(damping_coeff: float, frequency_damped: float) -> Optional[float]:
    """
    Berechnet den Dämpfungsgrad auf Basis von Dämpfungskonstante und gedämpfter Frequenz.

    Formel: ζ = d / f_d

    Args:
        damping_coeff (float): Dämpfungskonstante
        frequency_damped (float): Gedämpfte Frequenz [Hz]

    Returns:
        Optional[float]: Dämpfungsgrad (verhältnislos) oder None bei Fehler
    """
    try:
        return damping_coeff / frequency_damped
    except Exception as e:
        logger.warning(f"Could not calculate damping_ratio: {e}")
        return None
