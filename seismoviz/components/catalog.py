import pandas as pd

from seismoviz.analysis import Analyzer, GeoAnalyzer
from seismoviz.internal.decorators import sync_metadata
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

        self._analyzer = Analyzer(self)
        self._geo_analyzer = GeoAnalyzer(self)
    
    # Operations
    @sync_metadata(Analyzer, 'filter')
    def filter(self, **kwargs):
        return self._analyzer.filter(**kwargs)

    @sync_metadata(Analyzer, 'sort')
    def sort(self, **kwargs):
        return self._analyzer.sort(**kwargs)

    @sync_metadata(Analyzer, 'deduplicate_events')
    def deduplicate_events(self):
        return self._analyzer.deduplicate_events()

    # Geographical analysis
    @sync_metadata(GeoAnalyzer, 'plot_map')
    def plot_map(self, **kwargs) -> None:
        self._geo_analyzer.plot_map(**kwargs)

    @sync_metadata(GeoAnalyzer, 'plot_space_time')
    def plot_space_time(self, **kwargs) -> None:
        self._geo_analyzer.plot_space_time(**kwargs)

    @sync_metadata(GeoAnalyzer, 'plot_on_section')
    def plot_on_section(self, **kwargs) -> None:
        self._geo_analyzer.plot_on_section(**kwargs)

    # Magnitude analysis
    @sync_metadata(Analyzer, 'magnitude_time')
    def magnitude_time(self, **kwargs) -> None:
        self._analyzer.magnitude_time(**kwargs)
    
    @sync_metadata(Analyzer, 'fmd')
    def fmd(self, **kwargs):
        self._analyzer.fmd(**kwargs)
    
    @sync_metadata(Analyzer, 'b_value')
    def b_value(self, **kwargs):
        return self._analyzer.b_value(**kwargs)

    @sync_metadata(Analyzer, 'b_value_over_time')
    def b_value_over_time(self, **kwargs):
        return self._analyzer.b_value_over_time(**kwargs)

    # Statistical analysis
    @sync_metadata(Analyzer, 'event_timeline')
    def event_timeline(self, **kwargs) -> None:
        self._analyzer.event_timeline(**kwargs)

    @sync_metadata(Analyzer, 'interevent_time')
    def interevent_time(self, **kwargs):
        self._analyzer.interevent_time(**kwargs)

    @sync_metadata(Analyzer, 'cov')
    def cov(self, **kwargs):
        self._analyzer.cov(**kwargs)