import pandas as pd

from seismoviz.internal.decorators import sync_signature
from seismoviz.components.analysis.b_value import BValueCalculator
from seismoviz.internal.mixins import DunderMethodMixin, GeospatialMixin
from seismoviz.components.plotters.cat_plotter import CatalogPlotter, SubCatalogPlotter


class Catalog(GeospatialMixin, DunderMethodMixin):
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
            `attribute=('criteria', value)` or 
            `attribute=('criteria', [value1, value2])` for range criteria. 
            This allows for intuitive and flexible filtering based on the 
            attributes of seismic events.

        Parameters
        ----------
        **kwargs : dict
            Arbitrary keyword arguments representing the filtering 
            conditions. Each key is an attribute name of the seismic events 
            (e.g., 'mag', 'lat'), and the value is a tuple specifying the 
            criteria ('greater', 'lower', 'between', 'outside') and the 
            comparison value(s).

        Returns
        -------
        Catalog
            A new instance of Catalog containing the filtered subset of 
            seismic events.

        Examples
        --------
        To filter the catalog for events with magnitude greater than 4.5 
        and depth between the range [10, 50]:

        .. code-block:: python

            filtered_catalog = catalog.filter(
                mag=('greater', 4.5),
                depth=('between', [10, 50])
            )

        This would return a new `Catalog` instance containing only the 
        events that match the specified criteria.

        Raises
        ------
        ValueError
            If an invalid criteria is provided or if the value format does 
            not match the criteria requirements.

        .. note::
            The filtering operation is cumulative for multiple conditions. 
            For example, specifying conditions for both 'magnitude' and 
            'depth' will filter events that satisfy both conditions 
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
            The attribute of the seismic events to sort by (e.g., 'time', 
            'magnitude').

        ascending : bool, optional
            Determines the sorting order. If True (default), sorts in 
            ascending order; if False, in descending order.

        Returns
        -------
        Catalog
            The Catalog instance itself, modified to reflect the sorted 
            order.
        """
        return self.data.sort_values(by=by, ascending=ascending)

    def deduplicate_events(self) -> 'Catalog':
        """
        Removes duplicate seismic events based on longitude, latitude, 
        depth, and time, returning a new Catalog instance.

        Returns
        -------
        Catalog
            A new Catalog instance without duplicate events.
        """
        return self.data.drop_duplicates(subset=['lon', 'lat', 'depth', 'time'])

    @sync_signature('_plotter', 'plot_map')
    def plot_map(self, **kwargs) -> None:
        """
        Visualizes seismic events on a map.
        
        Parameters
        ----------
        highlight_mag : int, optional
            If specified, highlights all seismic events with a magnitude 
            greater than this value by plotting them as stars.

        color_by : str, optional
            Specifies the column in the DataFrame used to color the seismic 
            events. Common options include 'magnitude', 'time', or 'depth'. 
            If not provided, a default color is used.

        cmap : str, optional
            The colormap to use when coloring the events based on the 
            `color_by` column. Default is 'jet'.

        title : str, optional
            The title to be displayed above the map. If not provided, the 
            map will have no title.

        size : float, optional
            The size of the markers used to represent seismic events on 
            the map. Default is 0.5.

        color : str, optional
            The color used to fill the seismic event markers. Default is 
            'lightgrey'.

        edgecolor : str, optional
            The color used for the edges of the seismic event markers. 
            Default is 'grey'.

        alpha : float, optional
            The transparency level of the markers. A value between 0 and 1, 
            where 1 is fully opaque and 0 is fully transparent. Default is 0.5.

        legend : str, optional
            Text for the legend describing the plotted seismic events. If 
            None, no legend is displayed.

        xlim : Tuple[float, float], optional
            A tuple specifying the minimum and maximum longitude values to 
            set the map extent horizontally. If not provided, the extent 
            will be set automatically based on the data.

        ylim : Tuple[float, float], optional
            A tuple specifying the minimum and maximum latitude values to 
            set the map extent vertically. If not provided, the extent 
            will be set automatically based on the data.

        inset : bool, optional
            Determines whether to include an inset map showing a broader 
            geographic context. Defaults to True.

        inset_buffer : float, optional
            A factor that enlarges the buffer area around the selection 
            shape in the inset, enhancing visibility and context. The 
            default value is 3.

        bounds_res : str, optional
            The resolution for the geographical boundaries (such as 
            coastlines and borders) on the map. Common values are '10m', 
            '50m', or '110m', with '10m' being the most detailed and 
            '110m' the least.

        bmap_res : int, optional
            The zoom level or resolution for the underlying map image 
            (e.g., satellite or terrain map). A higher value provides a 
            more detailed map image, with typical values ranging from 1 
            (very coarse) to 12 (very detailed).

        save_figure : bool, optional
            If set to True, the function saves the generated plots using 
            the provided base name and file extension. The default is False.

        save_name : str, optional
            The base name used for saving figures when `save_figure` is True. 
            It serves as the prefix for file names. The default base name 
            is 'map'.

        save_extension : str, optional
            The file extension to use when saving figures, such as 'jpg', 
            'png', etc. The default extension is 'jpg'.
        """
        self._plotter.plot_map(**kwargs)

    @sync_signature('_plotter', 'plot_magnitude_time')
    def plot_magnitude_time(self, **kwargs) -> None:
        """
        Plots seismic event magnitudes over time.

        Parameters
        ----------
        save_figure : bool, optional
            If set to True, the function saves the generated plots using 
            the provided base name and file extension. The default is False.

        save_name : str, optional
            The base name used for saving figures when `save_figure` is True. 
            It serves as the prefix for file names. The default base name 
            is 'section'.

        save_extension : str, optional
            The file extension to use when saving figures, such as 'jpg', 
            'png', etc. The default extension is 'jpg'.
        """
        self._plotter.plot_magnitude_time(**kwargs)

    @sync_signature('_plotter', 'plot_event_timeline')
    def plot_event_timeline(self, **kwargs) -> None:
        """
        Plots a timeline of seismic events to visualize the cumulative 
        number of events over time.
        
        Parameters
        ----------
        save_figure : bool, optional
            If set to True, the function saves the generated plots using 
            the provided base name and file extension. The default is False.

        save_name : str, optional
            The base name used for saving figures when `save_figure` is True. 
            It serves as the prefix for file names. The default base name 
            is 'section'.

        save_extension : str, optional
            The file extension to use when saving figures, such as 'jpg', 
            'png', etc. The default extension is 'jpg'.
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
            If set to True, the function saves the generated plots using 
            the provided base name and file extension. The default is False.

        save_name : str, optional
            The base name used for saving figures when `save_figure` is True. 
            It serves as the prefix for file names. The default base name 
            is 'map'.

        save_extension : str, optional
            The file extension to use when saving figures, such as 'jpg', 
            'png', etc. The default extension is 'jpg'.
        """
        self._plotter.plot_attribute_distributions(**kwargs)

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
            Column name used to color points by a specific attribute. If None, 
            uses a fixed color.

        cmap : str, optional
            Colormap to use when coloring points by an attribute. Default is 'jet'.

        hl_ms : int, optional
            Magnitude threshold for highlighting large seismic events. Default is None.

        hl_size : float, optional
            The size of the highlighted events. Default is 200.

        hl_marker : str, optional
            Marker style for highlighted events. Default is '*'.

        hl_color : str, optional
            Color for highlighted seismic events. Default is 'red'.

        hl_edgecolor : str, optional
            Edge color for highlighted events. Default is 'darkred'.

        size : float or str, optional
            Size of the points or the name of the column to use for size scaling. 
            Default is 10.

        size_scale_factor : tuple[float, float], optional
            Scaling factors (base, exponent) for the point sizes. Default is (1, 2).

        color : str, optional
            Default color for the points if `color_by` is None. Default is 'grey'.

        edgecolor : str, optional
            Color for the edges of the points. Default is 'black'.

        alpha : float, optional
            Transparency level of the points. Default is 0.75.

        xlim : tuple of str, optional
            Time limits for the x-axis as start and end date strings. Default is None.

        ylim : tuple[float, float], optional
            Limits for the y-axis (distance from center). Default is None.

        legend : str, optional
            Label for the points. Default is None.

        legend_loc : str, optional
            Location for the legend. Default is 'lower right'.

        size_legend : bool, optional
            If True, includes a legend for point sizes. Default is False.

        size_legend_loc : str, optional
            Location for the size legend. Default is 'upper right'.

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is (10, 5).

        save_figure : bool, optional
            If True, saves the figure. Default is False.

        save_name : str, optional
            Base name for the saved figure. Default is 'map'.

        save_extension : str, optional
            File extension for the saved figure. Default is 'jpg'.
        """
        self._plotter.plot_space_time(**kwargs)
    
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
            If True, plots the FMD. Default is True.

        save_figure : bool, optional
            If True, saves the figure when `plot` is True. Default is False.

        save_name : str, optional
            The base name for saving the figure if `save_figure` is True. Default is 'fmd'.

        save_extension : str, optional
            The file extension for the saved figure. Default is 'jpg'.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            - bins : np.ndarray
                Array of magnitude bin edges.
            - events_per_bin : np.ndarray
                Array with the number of events in each magnitude bin.
            - cumulative_events : np.ndarray
                Array with the cumulative number of events for magnitudes greater than 
                or equal to each bin.

        """
        self._bvc.fmd(**kwargs, plot=True)
    
    def estimate_b_value(self, bin_size: float, mc: str | float, **kwargs):
        """
        Estimates the b-value for seismic events, a measure of earthquake 
        frequency-magnitude distribution, and calculates the associated uncertainties.

        Parameters
        ----------
        bin_size : float
            The size of each magnitude bin for calculating frequency-magnitude 
            distribution.

        mc : str or float
            The completeness magnitude (threshold), above which the b-value 
            estimation is considered valid.

        plot : bool, optional
            If True, plots the frequency-magnitude distribution with the 
            calculated b-value curve. Default is True.

        plot_uncertainty : str, optional
            Type of uncertainty to display in the plot. Options are 'shi_bolt' 
            for Shi and Bolt uncertainty and 'aki' for Aki uncertainty. Default is 'shi_bolt'.

        save_figure : bool, optional
            If True, saves the plot. Default is False.

        save_name : str, optional
            Base name for the saved figure, if `save_figure` is True. Default is 'b-value'.

        save_extension : str, optional
            File extension for the saved figure. Default is 'jpg'.

        Returns
        -------
        tuple[float, float, float, float]
            - a_value : float
                The a-value, representing the logarithmic scale of the seismicity rate.
            - b_value : float
                The b-value, indicating the relative occurrence of large and small 
                earthquakes.
            - aki_uncertainty : float
                The Aki uncertainty in the b-value estimation.
            - shi_bolt_uncertainty : float
                The Shi and Bolt uncertainty in the b-value estimation.
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