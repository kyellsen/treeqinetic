from typing import Optional

from kj_core.core_config import CoreConfig


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
        initial_params = {
            "initial_amplitude": 170,
            "damping_coeff": 0.3,
            "angular_frequency": 0.44,
            "phase_angle": 0,
            "y_shift": 0,
            "x_shift": 0
        }

        param_bounds = {
            "initial_amplitude": (130, 180),
            "damping_coeff": (0.1, 1.4),
            "angular_frequency": (0.3, 0.7),
            "phase_angle": (-0.5, 1.8),
            "y_shift": (-50, 50),
            "x_shift": (-2, 2)
        }

        metrics_warning = {
            "mse": (0, 1000),
            "mae": (0, 20),
            "rmse": (0, 30),
            "r2": (-10, 10)
        }

        error = 1000
