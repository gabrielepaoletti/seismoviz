import pandas as pd

from seismoviz.analysis import Analyzer, GeoAnalyzer
from seismoviz.internal.decorators import sync_methods
from seismoviz.internal.mixins import DunderMethodMixin, GeospatialMixin


@sync_methods([Analyzer, GeoAnalyzer])
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
    def filter(self, **kwargs):
        return self._analyzer.filter(**kwargs)

    def sort(self, **kwargs):
        return self._analyzer.sort(**kwargs)

    def deduplicate_events(self):
        return self._analyzer.deduplicate_events()

    # Geographical analysis
    def plot_map(self, **kwargs) -> None:
        self._geo_analyzer.plot_map(**kwargs)

    def plot_space_time(self, **kwargs) -> None:
        self._geo_analyzer.plot_space_time(**kwargs)

    def plot_on_section(self, **kwargs) -> None:
        self._geo_analyzer.plot_on_section(**kwargs)

    # Magnitude analysis
    def magnitude_time(self, **kwargs) -> None:
        self._analyzer.magnitude_time(**kwargs)
    
    def fmd(self, **kwargs):
        self._analyzer.fmd(**kwargs)
    
    def b_value(self, **kwargs):
        return self._analyzer.b_value(**kwargs)

    def b_value_over_time(self, **kwargs):
        return self._analyzer.b_value_over_time(**kwargs)

    # Statistical analysis
    def event_timeline(self, **kwargs) -> None:
        self._analyzer.event_timeline(**kwargs)

    def interevent_time(self, **kwargs):
        self._analyzer.interevent_time(**kwargs)

    def cov(self, **kwargs):
        self._analyzer.cov(**kwargs)

    def fit_omori(self, **kwargs):
        self._analyzer.fit_omori(**kwargs)