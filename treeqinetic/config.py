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
        param_labels = ["initial_amplitude", "damping_coeff", "angular_frequency", "phase_angle", "y_shift"]
        metrics_labels = ["mse", "mae", "rmse", "r2"]

        # Definiert die Anfangswerte für jeden Parameter
        initial_param_values = [170, 0.3, 0.44, 0, 0]
        # Definiert die Grenzwerte für jeden Parameter
        bounds_values = [(130, 180), (0.1, 1.4), (0.3, 0.7), (-0.5, 1.8), (-50, 50)]
        # Definiere die Metrics Warnings
        error = 1000
        metrics_warning_values = [(0, error), (0, 20), (0, 30), (-10, 10)]
