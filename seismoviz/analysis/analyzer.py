from seismoviz.analysis import *


class Analyzer(
    Operations, MagnitudeAnalysis, StatisticalAnalysis
):
    """
    Combines analysis modules into a single interface for simplicity.
    """
    def __init__(self, instance: object):
        Operations.__init__(self, instance)
        MagnitudeAnalysis.__init__(self, instance)
        StatisticalAnalysis.__init__(self, instance)


class GeoAnalyzer(
    GeoCatalog, GeoSection
):
    """
    Combines geographical analysis modules into a single interface for simplicity.
    """
    def __init__(self, instance: object):
        from seismoviz.components import Catalog, CrossSection

        if isinstance(instance, Catalog):
            GeoCatalog.__init__(self, instance)
        elif isinstance(instance, CrossSection):
            GeoSection.__init__(self, instance)
        else:
            raise TypeError(
                f"Unsupported type for instance: {type(instance).__name__}. "
                "Expected Catalog, CrossSection, or SubCatalog."
            )