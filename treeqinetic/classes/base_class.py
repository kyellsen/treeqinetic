from kj_core import get_logger
from kj_core.classes.core_base_class import CoreBaseClass

import treeqinetic

logger = get_logger(__name__)


class BaseClass(CoreBaseClass):
    """
    Base class built upon CoreBaseClass, using specific managers from treemotion.
    """

    def __init__(self):
        # Es wird angenommen, dass treemotion.CONFIG, usw. bereits initialisiert wurden
        # Initialisiere CoreBaseClass mit treemotion-Managern
        super().__init__(config=treeqinetic.CONFIG,
                         plot_manager=treeqinetic.PLOT_MANAGER)

