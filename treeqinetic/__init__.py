from typing import Tuple, Optional
from kj_logger import get_logger, LogManager, LOG_MANAGER
from kj_core import PlotManager
from .classes import Series, Sensor, Measurement, Oscillation
from .config import Config

from kj_logger import get_logger

CONFIG = None
PLOT_MANAGER = None


def setup(working_directory: Optional[str] = None, log_level: str = "info",
          safe_logs_to_file: bool = True) -> Tuple[Config, LogManager, PlotManager]:
    """
    Set up the treeqinetic package with specific configurations.

    Parameters:
        working_directory (str, optional): Path to the working directory.
        log_level (str, optional): Logging level.
        safe_logs_to_file
    """
    global CONFIG, PLOT_MANAGER

    LOG_MANAGER.update_config(working_directory, log_level, safe_logs_to_file)

    CONFIG = Config(working_directory)

    name = CONFIG.package_name
    name_s = CONFIG.package_name_short

    logger = get_logger(__name__)
    logger.info(f"{name_s}: Setup {name} package!")

    PLOT_MANAGER = PlotManager(CONFIG)

    logger.info(f"{name_s}: {name} setup completed.")

    return CONFIG, LOG_MANAGER, PLOT_MANAGER
