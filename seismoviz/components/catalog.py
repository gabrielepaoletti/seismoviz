import pandas as pd

from seismoviz.internal.decorators import sync_signature
from seismoviz.components.analysis.b_value import BValueCalculator
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

    _bvc : BValueCalculator
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
        self._bvc = BValueCalculator(self)

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

    @sync_signature('_plotter', 'plot_map')
    def plot_map(self, **kwargs) -> None:
        """
        Visualizes seismic events on a geographical map.

        Parameters
        ----------            
        color_by : str, optional
            Specifies the column in the DataFrame used to color the 
            seismic events. Default is ``None``, which applies a single color to 
            all points.

        cmap : str, optional
            The colormap to use for coloring events if ``color_by`` is specified. 
            Default is ``'jet'``.

        title : str, optional
            Title of the map. If ``None``, no title is displayed. Default is ``None``.

        hl_ms : int, optional
            If specified, highlights seismic events with a magnitude 
            greater than this value using different markers. Default is ``None``.

        hl_size : float, optional
            Size of the markers used for highlighted seismic events (if ``hl_ms`` 
            is specified). Default is 200.

        hl_marker : str, optional
            Marker style for highlighted events. Default is ``'*'``.

        hl_color : str, optional
            Color of the highlighted event markers. Default is ``'red'``.

        hl_edgecolor : str, optional
            Edge color for highlighted event markers. Default is ``'darkred'``.

        size : float or str, optional
            The size of the markers representing seismic events. If a string 
            is provided, it should refer to a column in the DataFrame to scale 
            point sizes proportionally. Default is 10.

        size_scale_factor : tuple[float, float], optional
            A tuple to scale marker sizes when ``size`` is based on a DataFrame 
            column. The first element scales the values, and the second element 
            raises them to a power. Default is ``(1, 3)``.

        color : str, optional
            Default color for event markers when ``color_by`` is ``None``. 
            Default is ``'grey'``.

        edgecolor : str, optional
            Edge color for event markers. Default is ``'black'``.

        alpha : float, optional
            Transparency level for markers, ranging from 0 (transparent) to 1 
            (opaque). Default is 0.75.

        legend : str, optional
            Text for the legend describing the seismic events. If ``None``, 
            no legend is displayed. Default is ``None``.

        legend_loc : str, optional
            Location of the legend for the seismic event markers. 
            Default is ``'lower left'``.

        size_legend : bool, optional
            If ``True``, displays a legend that explains marker sizes. Default is ``False``.
            
        size_legend_loc : str, optional
            Location of the size legend when ``size_legend`` is ``True``. Default is 
            ``'lower right'``.

        xlim : tuple[float, float], optional
            Longitude limits for the map's horizontal extent. If ``None``, 
            the limits are determined automatically based on the data. 
            Default is ``None``.

        ylim : tuple[float, float], optional
            Latitude limits for the map's vertical extent. If ``None``, 
            the limits are determined automatically based on the data. 
            Default is ``None``.

        terrain_cmap : str, optional
            The colormap to be applied to the terrain layer. Defaults to ``'gray_r'``.            

        terrain_style : str, optional
            The style of the terrain background for the map. Common values 
            include ``'satellite'``, ``'terrain'`` or ``'street'``.Defaults to 
            ``'satellite'``.

        terrain_alpha : float, optional
            The transparency level for the terrain layer, where 0 is fully 
            transparent and 1 is fully opaque. Defaults to 0.35.     

        projection : cartopy.crs projection, optional
            The map projection used to display the map. Defaults to 
            ``ccrs.Mercator()``.

        transform : cartopy.crs projection, optional
            The coordinate reference system of the data to be plotted. 
            Defaults to ``ccrs.PlateCarree()``.

        inset : bool, optional
            If ``True``, adds an inset map for broader geographic context. 
            Default is ``False``.

        inset_buffer : float, optional
            Scaling factor for the area surrounding the selection shape 
            in the inset map. Default is 3.

        bounds_res : str, optional
            Resolution of geographical boundaries (coastlines, borders) 
            on the map. Options are ``'10m'`` (highest resolution), ``'50m'``, 
            and ``'110m'`` (lowest resolution). Default is '50m'.

        bmap_res : int, optional
            Resolution level for the base map image (e.g., satellite or 
            terrain). Higher values provide more detail. Default is 12.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is ``'map'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A map with seismic events.
        
        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='global_seismic_catalog.csv')

            # Create a map showing earthquake locations
            catalog.plot_map(
                title='Global seismicity (M > 4.0)',
                color_by='depth',
                cmap='YlOrRd',
                size='mag',
                projection=ccrs.Robinson()
            )

        .. image:: https://imgur.com/0d6OA1L.jpg
            :align: center
        """
        self._plotter.plot_map(**kwargs)

    @sync_signature('_plotter', 'plot_space_time')
    def plot_space_time(self, **kwargs) -> None:
        """
        Plots the space-time distribution of seismic events along a specified 
        strike direction.

        Parameters
        ----------
        center : tuple[float, float]
            The (longitude, latitude) coordinates of the center point for distance calculation.

        strike : int
            The strike angle in degrees, measured clockwise from north. Defines 
            the direction along which distances are calculated.

        color_by : str, optional
            Specifies the column in the DataFrame used to color the 
            seismic events. Default is ``None``, which applies a single color to 
            all points.

        cmap : str, optional
            The colormap to use for coloring events if ``color_by`` is specified. 
            Default is ``'jet'``.

        hl_ms : int, optional
            If specified, highlights seismic events with a magnitude 
            greater than this value using different markers. Default is ``None``.

        hl_size : float, optional
            Size of the markers used for highlighted seismic events (if ``hl_ms`` 
            is specified). Default is 200.

        hl_marker : str, optional
            Marker style for highlighted events. Default is ``'*'``.

        hl_color : str, optional
            Color of the highlighted event markers. Default is ``'red'``.

        hl_edgecolor : str, optional
            Edge color for highlighted event markers. Default is ``'darkred'``.

        size : float or str, optional
            The size of the markers representing seismic events. If a string 
            is provided, it should refer to a column in the DataFrame to scale 
            point sizes proportionally. Default is 10.

        size_scale_factor : tuple[float, float], optional
            A tuple to scale marker sizes when ``size`` is based on a DataFrame 
            column. The first element scales the values, and the second element 
            raises them to a power. Default is ``(1, 2)``.

        color : str, optional
            Default color for event markers when ``color_by`` is ``None``. 
            Default is ``'grey'``.

        edgecolor : str, optional
            Edge color for event markers. Default is ``'black'``.

        alpha : float, optional
            Transparency level for markers, ranging from 0 (transparent) to 1 
            (opaque). Default is 0.75.

        legend : str, optional
            Text for the legend describing the seismic events. If ``None``, 
            no legend is displayed. Default is ``None``.

        legend_loc : str, optional
            Location of the legend for the seismic event markers. 
            Default is ``'lower left'``.

        size_legend : bool, optional
            If ``True``, displays a legend that explains marker sizes. Default is ``False``.
            
        size_legend_loc : str, optional
            Location of the size legend when ``size_legend`` is ``True``. Default is 
            ``'lower right'``.

        xlim : tuple[float, float], optional
            Time limits for the x-axis as start and end date strings. Default 
            is ``None``.

        ylim : tuple[float, float], optional
            Limits for the y-axis (distance from center). Default is ``None``.

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 5)``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is ``'space_time'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A space-time plot.

        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Plot space-time distribution
            catalog.plot_space_time(
                center=(13.12, 42.83),
                strike=155,
                hl_ms=5,
                size=0.5,
                color='black',
                alpha=0.25,
            )

        .. image:: https://imgur.com/AgrhmOt.jpg
            :align: center     
        """
        self._plotter.plot_space_time(**kwargs)

    @sync_signature('_plotter', 'plot_magnitude_time')
    def plot_magnitude_time(self, **kwargs) -> None:
        """
        Plots seismic event magnitudes over time.

        Parameters
        ----------            
        color_by : str, optional
            Specifies the column in the DataFrame used to color the 
            seismic events. Default is ``None``, which applies a single color to 
            all points.

        cmap : str, optional
            The colormap to use for coloring events if ``color_by`` is specified. 
            Default is ``'jet'``.

        size : float or str, optional
            The size of the markers representing seismic events. If a string 
            is provided, it should refer to a column in the DataFrame to scale 
            point sizes proportionally. Default is 10.

        size_scale_factor : tuple[float, float], optional
            A tuple to scale marker sizes when ``size`` is based on a DataFrame 
            column. The first element scales the values, and the second element 
            raises them to a power. Default is (1, 3).

        color : str, optional
            Default color for event markers when ``color_by`` is ``None``. 
            Default is ``'grey'``.

        edgecolor : str, optional
            Edge color for event markers. Default is ``'black'``.

        alpha : float, optional
            Transparency level for markers, ranging from 0 (transparent) to 1 
            (opaque). Default is 0.75.

        size_legend : bool, optional
            If ``True``, displays a legend that explains marker sizes. Default is ``False``.
            
        size_legend_loc : str, optional
            Location of the size legend when ``size_legend`` is ``True``. Default is 
            ``'upper right'``. 

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 5)``.
            
        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is ``'magnitude_time'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A magnitude-time plot.

        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Plot magnitude-time distribution
            catalog.plot_magnitude_time(
                color_by='depth',
                size='depth',
                cmap='YlOrRd',
            )
        
        .. image:: https://imgur.com/qYguHD1.jpg
            :align: center
        """
        self._plotter.plot_magnitude_time(**kwargs)

    @sync_signature('_plotter', 'plot_event_timeline')
    def plot_event_timeline(self, **kwargs) -> None:
        """
        Plots a timeline of seismic events to visualize the cumulative 
        number of events over time.  

        Parameters
        ----------  
        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 5)``.
            
        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is ``'event_timeline'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A event timeline plot.
        
        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Plot the event timeline
            catalog.plot_event_timeline()
        
        .. image:: https://imgur.com/FNnTzAV.jpg
            :align: center
        """
        self.plotter.plot_event_timeline(**kwargs)

    @sync_signature('_plotter', 'plot_attribute_distributions')
    def plot_attribute_distributions(self, **kwargs) -> None:
        """
        Visualizes the distribution of key attributes in the seismic event 
        catalog.
        
        Parameters
        ----------
        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is ``'attribute_distributions'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        None
            A plot showing the distribution of the main attributes of the 
            catalog.
        
        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Plot attribute distributions
            catalog.plot_attribute_distributions()
        
        .. image:: https://imgur.com/UfucKab.jpg
            :align: center
        """
        self._plotter.plot_attribute_distributions(**kwargs)
    
    @sync_signature('_bvc', 'fmd')
    def plot_fmd(self, **kwargs):
        """
        Calculates the frequency-magnitude distribution (FMD) for seismic events, 
        which represents the number of events in each magnitude bin.

        Parameters
        ----------
        bin_size : float
            The size of each magnitude bin.

        plot : bool, optional
            If ``True``, plots the FMD. Default is ``True``.

        return_values : bool, optional
            If ``True``, returns the calculated FMD values. Default is ``False``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is ``'fmd'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        .. warning::
            Values are returned only if ``return_values`` argument is set to 
            ``True``

        tuple[np.ndarray, np.ndarray, np.ndarray]
            - ``bins`` : np.ndarray
                Array of magnitude bin edges.
            - ``events_per_bin`` : np.ndarray
                Array with the number of events in each magnitude bin.
            - ``cumulative_events`` : np.ndarray
                Array with the cumulative number of events for magnitudes greater than 
                or equal to each bin.
        
        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Plot frequency-magnitude distributions
            catalog.plot_fmd(bin_size=0.1)
        
        .. image:: https://imgur.com/OWT7Pa5.jpg
            :align: center
        """
        self._bvc.fmd(**kwargs)
    
    def estimate_b_value(self, bin_size: float, mc: str | float, **kwargs):
        """
        Estimates the b-value for seismic events, and calculates the associated uncertainties.

        Parameters
        ----------
        bin_size : float
            The size of each magnitude bin for calculating frequency-magnitude 
            distribution.

        mc : str or float
            The completeness magnitude (threshold), above which the b-value 
            estimation is considered valid.

        plot : bool, optional
            If ``True``, plots the frequency-magnitude distribution with the 
            calculated b-value curve. Default is ``True``.

        plot_uncertainty : str, optional
            Type of uncertainty to display in the plot. Options are ``'shi_bolt'`` 
            for Shi and Bolt uncertainty and ``'aki'`` for Aki uncertainty. Default is ``'shi_bolt'``.

        return_values : bool, optional
            If ``True``, returns the calculated values. Default is ``False``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is ``'b-value'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default 
            is ``'jpg'``.

        Returns
        -------
        .. warning::
            Values are returned only if ``return_values`` argument is set to 
            ``True``

        tuple[float, float, float, float]
            - ``a_value`` : float
                The a-value, representing the logarithmic scale of the seismicity rate.
            - ``b_value`` : float
                The b-value, indicating the relative occurrence of large and small 
                earthquakes.
            - ``aki_uncertainty`` : float
                The Aki uncertainty in the b-value estimation.
            - ``shi_bolt_uncertainty`` : float
                The Shi and Bolt uncertainty in the b-value estimation.
        
        Raise
        -----
        ValueError
            If the selected Mc type or value is not valid.

        Examples
        --------
        .. code-block:: python

            import seismoviz as sv

            # Read the catalog from a file
            catalog = sv.read_catalog(path='local_seismic_catalog.csv')

            # Estimate b-value and plot it
            catalog.estimate_b_value(
                bin_size=0.1,
                mc='maxc',
                plot=True,
                return_values=False
            )
        
        .. image:: https://imgur.com/7vLgHh3.jpg
            :align: center
        """
        if mc == 'maxc':
            mc_maxc = self._bvc._maxc(bin_size=bin_size)
            return self._bvc.estimate_b_value(bin_size=bin_size, mc=mc_maxc, **kwargs)
        elif isinstance(mc, int) or isinstance(mc, float):
            return self._bvc.estimate_b_value(bin_size=bin_size, mc=mc, **kwargs)
        else:
            raise ValueError('Mc value is not valid.')


class SubCatalog(Catalog):
    def __init__(self, data: pd.DataFrame, selected_from: str) -> None:
        super().__init__(data)
        
        self.selected_from = selected_from
        self._sc_plotter = SubCatalogPlotter(self)
    
    @sync_signature('_sc_plotter', 'plot_on_section')
    def plot_on_section(self, **kwargs):
        self._sc_plotter.plot_on_section(**kwargs)