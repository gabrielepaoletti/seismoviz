import holoviews as hv
from holoviews import streams, opts

from seismoviz.components import Catalog


class BaseSelector:
    """
    Base class for handling plotting and selection of seismic event data.
    """

    def __init__(self) -> None:
        self.sd = None

        hv.extension('bokeh', logo=False)
        hv.opts.defaults(
            opts.Scatter(size=1, color='black')
        )

    @property
    def data(self):
        """
        Data to be plotted.

        Raises
        ------
        NotImplementedError
            If not implemented in a subclass.
        """
        raise NotImplementedError(
            "Subclasses must define the 'data' property."
        )

    def _selection_callback(self, index: list[int]) -> None:
        """Handle point selection events."""
        if index:
            self.sd = self.data.iloc[index]

    def _plot(self, **kwargs) -> hv.Scatter:
        """Create the Holoviews scatter plot."""
        raise NotImplementedError(
            "Subclasses must implement the '_plot' method."
        )

    def select(self, **kwargs) -> None:
        """
        Set up the plot and initialize the selection stream.

        Parameters
        ----------
        **kwargs : dict, optional
            Additional keyword arguments for the plot.
        """
        scatter = self._plot(**kwargs)
        selection = streams.Selection1D(source=scatter)
        selection.add_subscriber(self._selection_callback)
        hv.output(scatter)


class CatalogSelector(BaseSelector):
    """
    Selector for plotting seismic catalog data on a map and handling
    selection.
    """

    def __init__(self, catalog: object) -> None:
        self.ct = catalog
        super().__init__()

    @property
    def data(self):
        """
        Return the catalog data.

        Returns
        -------
        pandas.DataFrame
            Seismic event data with 'lon' and 'lat' columns.
        """
        return self.ct.data

    def _set_bounds(self, plot, element) -> None:
        """Set x and y range bounds for the map plot."""
        plot.state.x_range.bounds = (
            self.data.lon.min(), self.data.lon.max()
        )
        plot.state.y_range.bounds = (
            self.data.lat.min(), self.data.lat.max()
        )

    def _plot(self, **kwargs) -> hv.Scatter:
        """Create a scatter plot of the seismic catalog data."""
        scatter = hv.Scatter(
            self.data,
            kdims='lon',
            vdims='lat',
        ).opts(
            active_tools=['lasso_select', 'wheel_zoom', 'pan'],
            tools=['lasso_select', 'box_select'],
            aspect='equal',
            xlabel='Longitude [°]',
            ylabel='Latitude [°]',
            frame_width=800,
            xlim=(self.data.lon.min(), self.data.lon.max()),
            ylim=(self.data.lat.min(), self.data.lat.max()),
            framewise=False,
            hooks=[self._set_bounds],
            **kwargs
        )
        return scatter


class CrossSectionSelector(BaseSelector):
    """
    Selector for plotting seismic cross-section data and handling
    selection.
    """

    def __init__(self, cross_section: object) -> None:
        self.cs = cross_section
        super().__init__()

    @property
    def data(self):
        """
        Return the cross-section data.

        Returns
        -------
        pandas.DataFrame
            Seismic event data.
        """
        return self.cs.data

    def _set_bounds(self, plot, element) -> None:
        """Set x and y range bounds for the cross-section plot."""
        plot.state.x_range.bounds = (
            -self.cs.map_length / 2, self.cs.map_length / 2
        )
        plot.state.y_range.bounds = (
            self.cs.depth_range[0], self.cs.depth_range[1]
        )

    def _plot(self, **kwargs) -> hv.Scatter:
        """Create a scatter plot of the cross-section data."""
        scatter = hv.Scatter(
            self.data,
            kdims='on_section_coords',
            vdims='depth',
        ).opts(
            active_tools=['lasso_select', 'wheel_zoom', 'pan'],
            tools=['lasso_select', 'box_select'],
            aspect='equal',
            xlabel='Distance from center [km]',
            ylabel='Depth [km]',
            invert_yaxis=True,
            frame_width=min(self.cs.map_length * 30, 1000),
            xlim=(-self.cs.map_length / 2, self.cs.map_length / 2),
            ylim=(self.cs.depth_range[0], self.cs.depth_range[1]),
            framewise=False,
            hooks=[self._set_bounds],
            **kwargs
        )
        return scatter


class CustomSelector(BaseSelector):
    """
    Selector for custom plotting and selection of seismic event data.
    """

    def __init__(self, instance: object, x: str, y: str) -> None:
        self.it = instance
        self.x = x
        self.y = y
        super().__init__()

    @property
    def data(self):
        """
        Return the data from the provided instance.

        Returns
        -------
        pandas.DataFrame
            Seismic event data.
        """
        return self.it.data

    def _set_bounds(self, plot, element) -> None:
        """Set x and y range bounds for the custom plot."""
        plot.state.x_range.bounds = (
            self.data[self.x].min(), self.data[self.x].max()
        )  
        plot.state.y_range.bounds = (
            self.data[self.y].min(), self.data[self.y].max()
        )

    def _plot(self, **kwargs) -> hv.Scatter:
        """Create a custom scatter plot using specified x and y."""
        scatter = hv.Scatter(
            self.data,
            kdims=self.x,
            vdims=self.y,
        ).opts(
            active_tools=['lasso_select', 'wheel_zoom', 'pan'],
            tools=['lasso_select', 'box_select'],
            xlabel=self.x.capitalize(),
            ylabel=self.y.capitalize(),
            frame_width=1000,
            xlim=(self.data[self.x].min(), self.data[self.x].max()),
            ylim=(self.data[self.y].min(), self.data[self.y].max()),
            framewise=False,
            hooks=[self._set_bounds],
            **kwargs
        )
        return scatter


class SelectionWrapper:
    """
    Base wrapper class, centralizes common functionality for the interactive
    selection of seismic events.
    """

    def __init__(self, selector, **kwargs) -> None:
        self._selector = selector
        self._selector.select(**kwargs)

    def confirm_selection(self) -> Catalog:
        """
        Confirm the interactive selection and return the selected events.

        Returns
        -------
        Catalog
            A ``Catalog`` object containing the selected data.
        """
        return Catalog(data=self._selector.sd)


class MapSelection(SelectionWrapper):
    """
    Provides an interactive interface for selecting seismic events on a map.

    Parameters
    ----------
    catalog : Catalog
        The Catalog object containing seismic data and plotting
        configurations.

    **kwargs : dict, optional
        Additional keyword arguments passed to Holoviews `.opts()`
        method for customizing the scatter plot.
    """

    def __init__(self, catalog: Catalog, **kwargs) -> None:
        super().__init__(CatalogSelector(catalog), **kwargs)


class CrossSectionSelection(SelectionWrapper):
    """
    Provides an interactive interface for selecting seismic events on a
    cross-section.

    Parameters
    ----------
    cross_section : CrossSection
        The CrossSection object containing seismic data and plotting
        configurations.

    **kwargs : dict, optional
        Additional keyword arguments passed to Holoviews `.opts()`
        method for customizing the scatter plot.
    """

    def __init__(self, cross_section, **kwargs) -> None:
        super().__init__(CrossSectionSelector(cross_section), **kwargs)


class CustomSelection(SelectionWrapper):
    """
    Provides an interactive interface for selecting seismic events from a
    custom plot.

    Parameters
    ----------
    instance : type
        The instance object containing seismic data and plotting
        configurations.

    x : str
        The column name in the dataframe representing the x-axis variable
        for plotting.

    y : str
        The column name in the dataframe representing the y-axis variable
        for plotting.

    **kwargs : dict, optional
        Additional keyword arguments passed to Holoviews `.opts()`
        method for customizing the scatter plot.
    """

    def __init__(self, instance: type, x: str, y: str, **kwargs) -> None:
        super().__init__(CustomSelector(instance, x, y), **kwargs)