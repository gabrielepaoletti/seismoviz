import pandas as pd
import holoviews as hv
from holoviews import streams


class CrossSectionSelector:
    """
    A class to handle cross-section plotting and selection of seismic event 
    data in a Holoviews plot.

    Parameters
    ----------
    cross_section : type
        The class containing the cross-section data and plotting configurations.
    """

    def __init__(self, cross_section: type) -> None:
        self.cs = cross_section
        self.sd = None
        hv.extension('bokeh')

    def _set_bounds(self, plot, element) -> None:
        """
        Set the x and y range bounds for the cross-section plot.

        Parameters
        ----------
        plot : holoviews.plotting.bokeh.ElementPlot
            The plot object that contains the current state of the figure.

        element : holoviews.core.element.Element
            The element being plotted (usually the Scatter plot).
        """
        plot.state.x_range.bounds = (
            -self.cs.map_length / 2, self.cs.map_length / 2
        )
        plot.state.y_range.bounds = (
            self.cs.depth_range[0], self.cs.depth_range[1]
        )

    def _plot_section(self, size: float, color: str) -> hv.Scatter:
        """
        Create a Holoviews Scatter plot of the seismic data.

        Parameters
        ----------
        size : float
            The size of the points in the scatter plot.

        color : str
            The color of the points.

        Returns
        -------
        holoviews.core.element.Scatter
            A Scatter plot object configured with the cross-section data.
        """
        scatter = hv.Scatter(
            self.cs.data,
            kdims='on_section_coords',
            vdims='depth'
        ).opts(
            size=size,
            color=color,
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
            hooks=[self._set_bounds]
        )
        return scatter

    def _selection_callback(self, index: list[int]) -> None:
        """
        Callback function to handle point selection events.

        Parameters
        ----------
        index : list of int
            List of indices corresponding to selected points in the data.
        """
        if index:
            self.sd = self.cs.data.iloc[index]

    def select(self, size: float = 1, color: str = 'black') -> None:
        """
        Set up the cross-section plot and initialize the selection stream.

        Parameters
        ----------
        size : float, optional
            The size of the scatter plot points. Default is 1.

        color : str, optional
            The color of the points in the scatter plot. Default is 'black'.
        """
        scatter = self._plot_section(size=size, color=color)
        selection = streams.Selection1D(source=scatter)
        selection.add_subscriber(self._selection_callback)

        hv.output(scatter)