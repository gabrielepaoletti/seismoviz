import pandas as pd

from seismoviz.internal.decorators import sync_metadata
from seismoviz.components.analysis.operations import Operations
from seismoviz.components.analysis.magnitude import MagnitudeAnalyzer
from seismoviz.internal.mixins import DunderMethodMixin, GeospatialMixin
from seismoviz.components.plotters.cat_plotter import CatalogPlotter, SubCatalogPlotter


class Catalog(GeospatialMixin, DunderMethodMixin):
    """
    Represents a seismic event catalog.

    Attributes
    ----------
    data : pandas.DataFrame
        A DataFrame containing the seismic event data.

    _plotter : CatalogPlotter
        An internal object responsible for generating plots and visualizations 
        of the catalog.

    _mag : MagnitudeAnalyzer
        An internal object used for magnitude analysis.

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
        self._mag = MagnitudeAnalyzer(self.data.mag)

    @sync_metadata(Operations, 'filter')
    def filter(self, **kwargs):
        return Operations.filter(self, **kwargs)

    @sync_metadata(Operations, 'sort')
    def sort(self, by: str, ascending: bool = True):
        return Operations.sort(self, by=by, ascending=ascending)

    @sync_metadata(Operations, 'deduplicate_events')
    def deduplicate_events(self):
        return Operations.deduplicate_events(self)

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

    @sync_metadata(CatalogPlotter, 'plot_interevent_time')
    def plot_interevent_time(self, **kwargs) -> None:
        self._plotter.plot_interevent_time(**kwargs)
    
    @sync_metadata(MagnitudeAnalyzer, 'fmd')
    def fmd(self, **kwargs):
        self._mag.fmd(**kwargs)
    
    @sync_metadata(MagnitudeAnalyzer, 'estimate_b_value')
    def estimate_b_value(self, bin_size: float, mc: str | float, **kwargs):
        if mc == 'maxc':
            mc_maxc = self._mag._maxc(bin_size=bin_size)
            return self._mag.estimate_b_value(
                bin_size=bin_size, mc=mc_maxc, **kwargs
            )
        elif isinstance(mc, int) or isinstance(mc, float):
            return self._mag.estimate_b_value(
                bin_size=bin_size, mc=mc, **kwargs
            )
        else:
            raise ValueError('Mc value is not valid.')


class SubCatalog(Catalog):
    def __init__(self, data: pd.DataFrame, selected_from: str) -> None:
        super().__init__(data)
        
        self.selected_from = selected_from
        self._sc_plotter = SubCatalogPlotter(self)
    
    @sync_metadata(SubCatalogPlotter, 'plot_on_section')
    def plot_on_section(self, **kwargs):
        self._sc_plotter.plot_on_section(**kwargs)