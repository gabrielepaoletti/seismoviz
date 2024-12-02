import pandas as pd

from seismoviz.components.analysis import Analyzer
from seismoviz.internal.decorators import sync_metadata
from seismoviz.components.visualization import CatalogPlotter
from seismoviz.internal.mixins import DunderMethodMixin, GeospatialMixin


class Catalog(GeospatialMixin, DunderMethodMixin):
    """
    Represents a seismic event catalog.

    Attributes
    ----------
    data : pandas.DataFrame
        A DataFrame containing the seismic event data.

    Raises
    ------
    ValueError
        If the input DataFrame is missing required columns during initialization.

        .. warning::
            The input CSV file must contain the following columns: 
            ``lon``, ``lat``, ``time``, ``depth``, ``mag``, and ``id``.
    """
    def __init__(self, data: pd.DataFrame) -> None:
        missing = {'lon', 'lat', 'depth', 'time', 'mag'} - set(data.columns)
        if missing:
            raise ValueError(
                f"Missing required columns: {', '.join(missing)}. "
                f"You may have to add {'it' if len(missing) == 1 else 'them'} "
                "or rename the existing columns."
            )

        self.data = data
        super().__init__()

        self._plotter = CatalogPlotter(self)
        self._analyzer = Analyzer(self)

    @sync_metadata(Analyzer, 'filter')
    def filter(self, **kwargs):
        return self._analyzer.filter(**kwargs)

    @sync_metadata(Analyzer, 'sort')
    def sort(self, **kwargs):
        return self._analyzer.sort(**kwargs)

    @sync_metadata(Analyzer, 'deduplicate_events')
    def deduplicate_events(self):
        return self._analyzer.deduplicate_events()

    @sync_metadata(CatalogPlotter, 'plot_map')
    def plot_map(self, **kwargs) -> None:
        self._plotter.plot_map(**kwargs)

    @sync_metadata(CatalogPlotter, 'plot_space_time')
    def plot_space_time(self, **kwargs) -> None:
        self._plotter.plot_space_time(**kwargs)

    @sync_metadata(CatalogPlotter, 'plot_magnitude_time')
    def plot_magnitude_time(self, **kwargs) -> None:
        self._plotter.plot_magnitude_time(**kwargs)

    @sync_metadata(CatalogPlotter, 'plot_event_timeline')
    def plot_event_timeline(self, **kwargs) -> None:
        self._plotter.plot_event_timeline(**kwargs)

    @sync_metadata(CatalogPlotter, 'plot_attribute_distributions')
    def plot_attribute_distributions(self, **kwargs) -> None:
        self._plotter.plot_attribute_distributions(**kwargs)
    
    @sync_metadata(Analyzer, 'fmd')
    def fmd(self, **kwargs):
        self._analyzer.fmd(**kwargs)
    
    @sync_metadata(Analyzer, 'estimate_b_value')
    def estimate_b_value(self, bin_size: float, mc: str | float, **kwargs):
        if mc == 'maxc':
            mc_maxc = self._analyzer._maxc(bin_size=bin_size)
            return self._analyzer.estimate_b_value(
                bin_size=bin_size, mc=mc_maxc, **kwargs
            )
        elif isinstance(mc, int) or isinstance(mc, float):
            return self._analyzer.estimate_b_value(
                bin_size=bin_size, mc=mc, **kwargs
            )
        else:
            raise ValueError('Mc value is not valid.')

    @sync_metadata(Analyzer, 'interevent_time')
    def interevent_time(self, **kwargs):
        self._analyzer.interevent_time(**kwargs)

    @sync_metadata(Analyzer, 'cov')
    def cov(self, **kwargs):
        self._analyzer.cov(**kwargs)