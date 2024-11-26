import numpy as np
import matplotlib.pyplot as plt

from seismoviz.components.common import styling
from seismoviz.components.common.base_plotter import BasePlotter


class StatisticalAnalyzer:
    def __init__(self, catalog: type):
        self.ct = catalog
        self.bp = BasePlotter()

    def interevent_time(self):
        pass

    def coefficient_of_variation(self):
        pass