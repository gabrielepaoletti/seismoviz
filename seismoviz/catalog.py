import pandas as pd

from seismoviz.internal.decorators import sync_signature
from seismoviz.plotters.cat_plotter import CatalogPlotter
from seismoviz.internal.mixins import DunderMethodMixin, GeospatialMixin


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
                mw=('greater', 4.5),
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

    @sync_signature('plot_map', CatalogPlotter)
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

    @sync_signature('plot_magnitude_time', CatalogPlotter)
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

    @sync_signature('plot_event_timeline', CatalogPlotter)
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

    @sync_signature('plot_attribute_distributions', CatalogPlotter)
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
