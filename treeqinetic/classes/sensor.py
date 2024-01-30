from .base_class import BaseClass

from kj_logger import get_logger

logger = get_logger(__name__)


class Sensor(BaseClass):
    def __int__(self):
        super().__init__()
    pass
