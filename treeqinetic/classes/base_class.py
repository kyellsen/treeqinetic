import treeqinetic

from kj_core import get_logger
logger = get_logger(__name__)


class BaseClass:
    def __init__(self):
        managers_to_check = [treeqinetic.CONFIG, treeqinetic.PLOT_MANAGER]

        if any(manager is None for manager in managers_to_check):
            raise ValueError(f"The package has not been properly initialized. Please call the setup function first.")

        else:
            self.CONFIG = treeqinetic.CONFIG
            self.PLOT_MANAGER = treeqinetic.PLOT_MANAGER
