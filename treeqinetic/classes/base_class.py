from kj_core import get_logger
import treeqinetic

logger = get_logger(__name__)


class BaseClass():
    __abstract__ = True
    _config = None
    _plot_manager = None

    def __init__(self):
        super().__init__()

    @property
    def CONFIG(self):
        if self._config is None:
            self._config = treeqinetic.CONFIG
        return self._config

    @property
    def PLOT_MANAGER(self):
        if self._plot_manager is None:
            self._plot_manager = treeqinetic.PLOT_MANAGER
        return self._plot_manager
