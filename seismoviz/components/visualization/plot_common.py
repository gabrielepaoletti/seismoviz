import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seismoviz.components.visualization.utils.plot_utils as pu

from seismoviz.components.visualization.utils import styling
from seismoviz.components.analysis.statistical import StatisticalAnalysis


class CommonPlotter:
    """
    Provides common plotting methods for seismic data visualization.
    """

    def __init__(self, instance: type) -> None:
        self._instance = instance

    @property
    def data(self):
        """
        Dynamically fetches the data from the source instance.
        """
        return self._instance.data

    def plot_magnitude_time(
            self,
            color_by: str = None,
            cmap: str = 'jet',
            size: float | str = 10,
            size_scale_factor: tuple[float, float] = (1, 2),
            color: str = 'grey',
            edgecolor: str = 'black',
            alpha: float = 0.75,
            size_legend: bool = False,
            size_legend_loc: str = 'upper right',
            fig_size: tuple[float, float] = (10, 5),
            save_figure: bool = False,
            save_name: str = 'magnitude_time',
            save_extension: str = 'jpg'
    ) -> None:
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
            raises them to a power. Default is (1, 2).

        color : str, optional
            Default color for event markers when ``color_by`` is ``None``.
            Default is ``'grey'``.

        edgecolor : str, optional
            Edge color for event markers. Default is ``'black'``.

        alpha : float, optional
            Transparency level for markers, ranging from 0 (transparent) to 1
            (opaque). Default is 0.75.

        size_legend : bool, optional
            If ``True``, displays a legend that explains marker sizes. Default is
            ``False``.

        size_legend_loc : str, optional
            Location of the size legend when ``size_legend`` is ``True``. Default
            is ``'upper right'``.

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 5)``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is
            ``'magnitude_time'``.

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

            # Plot magnitude-time distribution
            instance.plot_magnitude_time(
                color_by='depth',
                size='depth',
                cmap='YlOrRd',
            )

        .. image:: https://imgur.com/qYguHD1.jpg
            :align: center
        """
        pu.set_style(styling.DEFAULT)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title('Magnitude-time distribution', fontweight='bold')

        plt_size = pu.process_size_parameter(size, self.data, size_scale_factor)

        if color_by:
            fig.set_figwidth(fig_size[0] + 2)
            pu.plot_with_colorbar(
                ax=ax,
                data=self.data,
                x='time',
                y='mag',
                color_by=color_by,
                cmap=cmap,
                size=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                cbar_orientation='vertical',
                cbar_shrink=1,
                cbar_aspect=30,
                cbar_pad=0.03
            )
        else:
            ax.scatter(
                x=self.data.time,
                y=self.data.mag,
                c=color,
                s=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                lw=0.25
            )

        if size_legend and isinstance(size, str):
            pu.create_size_legend(
                ax=ax,
                size=size,
                data=self.data,
                size_scale_factor=size_scale_factor,
                alpha=alpha,
                size_legend_loc=size_legend_loc
            )

        pu.format_x_axis_time(ax)
        ax.set_ylabel('Magnitude')
        ax.grid(True, alpha=0.25, axis='y', linestyle=':')

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()

    def plot_event_timeline(
            self,
            fig_size: tuple[float, float] = (10, 5),
            save_figure: bool = False,
            save_name: str = 'event_timeline',
            save_extension: str = 'jpg'
    ) -> None:
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
            Base name for the file if `save_figure` is ``True``. Default is
            ``'event_timeline'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default
            is ``'jpg'``.

        Returns
        -------
        None
            An event timeline plot.

        Examples
        --------
        .. code-block:: python

            # Plot the event timeline
            instance.plot_event_timeline()

        .. image:: https://imgur.com/FNnTzAV.jpg
            :align: center
        """
        pu.set_style(styling.DEFAULT)

        events_sorted = self.data.sort_values('time')
        time_data = pd.to_datetime(events_sorted['time'])

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title('Event timeline', fontweight='bold')
        ax.plot(time_data, np.arange(len(events_sorted)), color='black')

        pu.format_x_axis_time(ax)

        ax.set_ylabel('Cumulative no. of events')
        ax.set_ylim(0)
        ax.grid(True, axis='x', linestyle=':', linewidth=0.25)

        plt.tight_layout()
        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()

    def plot_attribute_distributions(
            self,
            fig_size: tuple[float, float] = (10, 6),
            save_figure: bool = False,
            save_name: str = 'attribute_distributions',
            save_extension: str = 'jpg'
    ) -> None:
        """
        Visualizes the distribution of key attributes in the seismic event
        data.

        Parameters
        ----------
        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 6)``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is
            ``'attribute_distributions'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default
            is ``'jpg'``.

        Returns
        -------
        None
            A plot showing the distribution of the main attributes of the
            data.

        Examples
        --------
        .. code-block:: python

            # Plot attribute distributions
            instance.plot_attribute_distributions()

        .. image:: https://imgur.com/UfucKab.jpg
            :align: center
        """
        pu.set_style(styling.DEFAULT)

        rows, cols = 2, 2
        fig, ax = plt.subplots(rows, cols, figsize=fig_size, sharey=True)
        plt.suptitle('Attribute distributions', fontsize=18, fontweight='bold')
        labels = {'lon': 'Longitude', 'lat': 'Latitude', 'mag': 'Magnitude', 'depth': 'Depth'}

        for i, (attribute, label) in enumerate(labels.items()):
            row, col = divmod(i, cols)
            ax[row, col].hist(
                self.data[attribute], bins=50, color='silver',
                edgecolor='black', linewidth=0.5
            )
            ax[row, col].set_xlabel(label)
            ax[row, col].set_ylabel('Number of events' if col == 0 else '')
            ax[row, col].grid(True, alpha=0.25, axis='y', linestyle=':')

        plt.tight_layout()
        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()
