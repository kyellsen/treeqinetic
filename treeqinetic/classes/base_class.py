from kj_core import get_logger

import treeqinetic

logger = get_logger(__name__)


class BaseClass:
    """
    Base class built upon CoreBaseClass, using specific managers from treemotion.
    """

    def __init__(self):
        self.CONFIG = treeqinetic.CONFIG
        self.PLOT_MANAGER = treeqinetic.PLOT_MANAGER
