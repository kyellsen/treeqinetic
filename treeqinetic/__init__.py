from typing import Tuple, Optional
from kj_core import PlotManager
from kj_core import log_manager, get_logger

import classes
from .config import Config

CONFIG = None
PLOT_MANAGER = None


def setup(working_directory: Optional[str] = None, log_level: Optional[str] = None) -> Tuple[Config, PlotManager]:
    """
    Set up the treeqinetic package with specific configurations.

    Parameters:
        working_directory (str, optional): Path to the working directory.
        log_level (str, optional): Logging level.
    """
    global CONFIG, PLOT_MANAGER

    CONFIG = Config(working_directory, log_level)
    log_manager.configure_logger(CONFIG)

    name = CONFIG.package_name
    name_s = CONFIG.package_name_short

    logger = get_logger(__name__)
    logger.info(f"{name_s}: Setup {name} package!")
    logger.info(f"{name_s}: CONFIG initialized: {CONFIG}")
    logger.info(f"{name_s}: LOGGER initialized")

    PLOT_MANAGER = PlotManager(CONFIG)
    logger.info(f"{name_s}: PLOT_MANAGER initialized: {PLOT_MANAGER}")

    logger.info(f"{name_s}: {name} setup completed.")

    return CONFIG, PLOT_MANAGER
