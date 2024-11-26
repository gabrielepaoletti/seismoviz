import pandas as pd

from seismoviz.internal.decorators import sync_metadata
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
        An internal object used for calculating the b-value and related seismic 
        analysis metrics.

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

    def filter(self, **kwargs) -> 'Catalog':
        """
        Filters the catalog based on multiple specified conditions, with 
        each condition passed as a keyword argument.

        .. note::
            Each keyword argument should be in the form 
            ``attribute=('criteria', value)`` or 
            ``attribute=('criteria', [value1, value2])`` for range criteria. 
            This allows for intuitive and flexible filtering based on the 
            attributes of seismic events.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments representing the filtering 
            conditions. Each key is an attribute name of the seismic events, 
            and the value is a tuple specifying the criteria (``'greater'``, 
            ``'lower'``, ``'between'``, ``'outside'``) and the comparison value(s).

        Returns
        -------
        Catalog
            A new instance of ``Catalog`` containing the filtered subset of 
            seismic events.

        Examples
        --------
        To filter the catalog for events with magnitude greater than 4.5, depth 
        between the range 10-50 km and occured before October 30, 2016:

        .. code-block:: python

            filtered_catalog = catalog.filter(
                mag=('greater', 4.5),
                depth=('between', [10, 50]),
                time=('lower', '2016-10-30')
            )

        This would return a new ``Catalog`` instance containing only the 
        events that match the specified criteria.

        Raises
        ------
        ValueError
            If an invalid criteria is provided or if the value format does 
            not match the criteria requirements.

        .. note::
            The filtering operation is cumulative for multiple conditions. 
            For example, specifying conditions for both ``'magnitude'`` and 
            ``'depth'`` will filter events that satisfy both conditions 
            simultaneously.
        """
        filtered_data = self.data

        for attribute, (criteria, value) in kwargs.items():
            if criteria == 'greater':
                filtered_data = filtered_data[filtered_data[attribute] > value]

            elif criteria == 'lower':
                filtered_data = filtered_data[filtered_data[attribute] < value]

            elif criteria == 'between':
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(
                        "Value must be a list of two numbers for 'between' criteria."
                    )
                filtered_data = filtered_data[
                    filtered_data[attribute].between(value[0], value[1])
                ]

            elif criteria == 'outside':
                if not isinstance(value, list) or len(value) != 2:
                    raise ValueError(
                        "Value must be a list of two numbers for 'outside' criteria."
                    )
                filtered_data = filtered_data[
                    ~filtered_data[attribute].between(value[0], value[1])
                ]

            else:
                raise ValueError(
                    f"Invalid criteria '{criteria}'. Choose from 'greater', 'lower', "
                    "'between', or 'outside'."
                )

        return Catalog(filtered_data)

    def sort(self, by: str, ascending: bool = True) -> 'Catalog':
        """
        Sorts the catalog by a specific attribute, either in ascending or 
        descending order.

        Parameters
        ----------
        by : str
            The attribute of the seismic events to sort by.

        ascending : bool, optional
            Determines the sorting order. If ``True`` (default), sorts in 
            ascending order; if ``False``, in descending order.

        Returns
        -------
        Catalog
            The ``Catalog`` instance itself, modified to reflect the sorted 
            order.

        Examples
        --------
        To sort the catalog by time in ascending order:

        .. code-block:: python

            sorted_catalog = catalog.sort(
                by='time',
                ascending=True
            )

        """
        return self.data.sort_values(by=by, ascending=ascending)

    def deduplicate_events(self) -> 'Catalog':
        """
        Removes duplicate seismic events.

        Returns
        -------
        Catalog
            A new ``Catalog`` instance without duplicate entries.

        Examples
        --------
        To remove duplicates inside the catalog:

        .. code-block:: python

            deduplicated_catalog = catalog.deduplicate_events()
        
        """
        return self.data.drop_duplicates(subset=['lon', 'lat', 'depth', 'time'])

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
        self.plotter.plot_event_timeline(**kwargs)

    @sync_metadata(CatalogPlotter, 'plot_attribute_distributions')
    def plot_attribute_distributions(self, **kwargs) -> None:
        self._plotter.plot_attribute_distributions(**kwargs)
    
    @sync_metadata(MagnitudeAnalyzer, 'fmd')
    def fmd(self, **kwargs):
        self._mag.fmd(**kwargs)
    
    @sync_metadata(MagnitudeAnalyzer, 'estimate_b_value')
    def estimate_b_value(self, bin_size: float, mc: str | float, **kwargs):
        if mc == 'maxc':
            mc_maxc = self._mag._maxc(bin_size=bin_size)
            return self._mag.estimate_b_value(bin_size=bin_size, mc=mc_maxc, **kwargs)
        elif isinstance(mc, int) or isinstance(mc, float):
            return self._mag.estimate_b_value(bin_size=bin_size, mc=mc, **kwargs)
        else:
            raise ValueError('Mc value is not valid.')


class SubCatalog(Catalog):
    def __init__(self, data: pd.DataFrame, selected_from: str) -> None:
        super().__init__(data)
        
        self.selected_from = selected_from
        self._sc_plotter = SubCatalogPlotter(self)
    
    def plot_on_section(self, **kwargs):
        self._sc_plotter.plot_on_section(**kwargs)