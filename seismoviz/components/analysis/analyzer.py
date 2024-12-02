from seismoviz.components.analysis import *


class Analyzer(
    Operations, MagnitudeAnalysis, StatisticalAnalysis
):
    """
    Combines analysis modules into a single interface for simplicity.
    """
    def __init__(self, instance):
        super().__init__(instance)