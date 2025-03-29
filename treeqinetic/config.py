from typing import Optional

from kj_core.core_config import CoreConfig
import numpy as np


class Config(CoreConfig):
    """
    Configuration class for the package, extending the core configuration.
    Provides package-specific settings and default values.
    """
    # Override default working directory specific
    package_name = "treeqinetic"
    package_name_short = "ptq"
    # Override default working directory specific
    default_working_directory = r"C:\kyellsen\006_Packages\treeqinetic\working_directory_ptq"

    def __init__(self, working_directory: Optional[str] = None):
        """
        Initializes the configuration settings, building upon the core configuration.
        """
        super().__init__(f"{working_directory}/{self.package_name_short}")

    class Oscillation:
        param_labels = ["initial_amplitude", "damping_coeff", "frequency_damped", "phase_angle", "y_shift", "x_shift"]

        initial_param = {
            "initial_amplitude": 170,
            "damping_coeff": 0.32,
            "frequency_damped": 0.44,
            "phase_angle": 0,
            "y_shift": 0,
            "x_shift": 0
        }

        param_bounds = {
            "initial_amplitude": (150, 250),
            "damping_coeff": (0.1, 1),
            "frequency_damped": (0.35, 0.58),
            "phase_angle": (-0.2, 0.2),
            "y_shift": (-60, 60),
            "x_shift": (-0.25, 0.75),
        }

        param_add_labels = ["frequency_undamped", "damping_ratio"]

        metrics_warning = {
            "pearson_r": (0.75, 1),
            "nrmse": (0, np.inf),
            "mae": (0, np.inf),
            "nmae": (0, 0.10)
        }

        metrics_to_plot = ['pearson_r', 'p_value', 'r2', 'nrmse', 'mae', 'nmae']

        # Neue Default-Optionen für scipy.optimize.minimize
        fit_options = {
            'maxiter': 1000,
            'ftol': 1e-8,
        }
        #error = 1000
