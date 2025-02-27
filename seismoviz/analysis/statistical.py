import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seismoviz.analysis.utils import styling
from seismoviz.analysis.utils import plot_utils as pu

from numpy.typing import ArrayLike

class StatisticalAnalysis:
    def __init__(self, instance: object):
        self._instance = instance
        self._disable_warnings()

    def event_timeline(
            self,
            ms_line: float = None,
            ms_line_color: str = 'red',
            ms_line_width: float = 1.5,
            ms_line_style: str = '-',
            ms_line_gradient : bool = True,
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
        ms_line : float, optional
            The magnitude threshold above which vertical lines will be added to 
            the plot. If ``None``, no vertical lines are added. Default is ``None``.

        ms_line_color : str, optional
            The color of the vertical lines. Accepts any Matplotlib-compatible 
            color string. Default is ``'red'``.

        ms_line_width : float, optional
            The thickness of the vertical lines. Default is 1.5.

        ms_line_style : str, optional
            The style of the vertical lines. Default is ``'-'``.

        ms_line_gradient : bool, optional
            If ``True``, the vertical lines will have a gradient effect, fading 
            from the pecified color to transparent along the y-axis. If ``False``, 
            the lines will be solid. Default is ``True``.

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
        """
        pu.set_style(styling.DEFAULT)

        events_sorted = self._instance.data.sort_values('time')
        time_data = pd.to_datetime(events_sorted['time'])

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title('Event timeline', fontweight='bold')
        ax.plot(time_data, np.arange(len(events_sorted)), color='black')
        pu.add_vertical_lines(
            ax=ax,
            data=self._instance.data,
            ms_line=ms_line,
            color=ms_line_color,
            linewidth=ms_line_width,
            linestyle=ms_line_style,
            gradient=ms_line_gradient
        )

        pu.format_x_axis_time(ax)

        ax.set_ylabel('Cumulative no. of events')
        ax.set_ylim(0)
        ax.grid(True, axis='x', linestyle=':', alpha=0.25)

        plt.tight_layout()
        if save_figure:
            pu.save_figure(save_name, save_extension)

        pu.reset_style()

    def interevent_time(self, unit='sec', plot: bool = True, **kwargs) -> None:
        """
        Calculates the inter-event time for sequential events in the instance.

        .. note::
            After executing this method, a new column named ``interevent_time`` 
            will be created in the `instance.data` DataFrame. This column will 
            contain the calculated inter-event times, making the data accessible 
            for further analysis or visualization.

        Parameters
        ----------
        unit : str, optional
            The time unit for the inter-event times. Supported values are:
            - ``'sec'``: seconds (default)
            - ``'min'``: minutes
            - ``'hour'``: hours
            - ``'day'``: days

        plot : bool, optional
            If ``True``, plots the interevent time distribution. Default is 
            ``True``.

        plot_vs : str, optional
            Specifies the column to plot inter-event times against. Default is
            ``'time'``.

        plot_event_timeline : bool, optional
            If ``True`` and ``plot_vs='time'``, adds a secondary y-axis (``twiny``)
            to plot a cumulative event timeline. The timeline represents the cumulative
            number of events over time. Default is ``True``.

        et_color : str, optional
            Specifies the color used for the secondary y-axis (event timeline axis), 
            including the ticks, labels, axis line, and the line representing 
            the cumulative number of events. Default is ``'red'``.

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
            raises them to a power. Default is ``(1, 2)``.

        yscale : str, optional
            Specifies the scale of the y-axis. Common options include ``'linear'``,
            ``'log'``, ``'symlog'``, and ``'logit'``. Default is ``'log'``.

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
            Location of the size legend when ``size_legend`` is ``True``.
            Default is ``'upper right'``.

        ms_line : float, optional
            The magnitude threshold above which vertical lines will be added to 
            the plot. If ``None``, no vertical lines are added. Default is ``None``.

        ms_line_color : str, optional
            The color of the vertical lines. Accepts any Matplotlib-compatible 
            color string. Default is ``'orange'``.

        ms_line_width : float, optional
            The thickness of the vertical lines. Default is 1.5.

        ms_line_style : str, optional
            The style of the vertical lines. Default is ``'-'``.

        ms_line_gradient : bool, optional
            If ``True``, the vertical lines will have a gradient effect, fading 
            from the pecified color to transparent along the y-axis. If ``False``, 
            the lines will be solid. Default is ``True``.    

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 5)``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is
            ``'interevent_time'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default
            is ``'jpg'``.

        Raises
        ------
        ValueError
            If an unsupported time unit is provided.

        ValueError
            If an invalid configuration is provided.
        """
        valid_units = {'sec': 1, 'min': 60, 'hour': 3600, 'day': 86400}
        if unit not in valid_units:
            raise ValueError(
                f"Unit '{unit}' not supported. Choose from {list(valid_units.keys())}."
            )
        
        self._instance.sort(by='time', ascending=True)
        
        time_diffs = self._instance.data['time'].diff().dt.total_seconds()
        self._instance.data['interevent_time'] = time_diffs / valid_units[unit]

        if plot:
            if kwargs.get('plot_cov', False):
                raise ValueError(
                    "To plot the COV, please use `instance.cov()` instead."
                )
            self._plot_interevent_time(**kwargs)

    def cov(self, window_size: int, plot: bool = True, **kwargs) -> None:
        """
        Calculates the coefficient of variation (COV) for the inter-event times 
        using a rolling window.

        .. note::
            After executing this method, a new column named ``cov`` will be 
            created in the `instance.data` DataFrame. This column will contain 
            the calculated inter-event times, making the data accessible for 
            further analysis or visualization.

        Parameters
        ----------
        window_size : int
            The size of the rolling window (in number of events) over which the 
            coefficient of variation is calculated.

        plot : bool, optional
            If ``True``, plots the interevent time distribution along with the COV. 
            Default is ``True``.

        plot_event_timeline : bool, optional
            If ``True`` and ``plot_vs='time'``, adds a secondary y-axis (``twiny``)
            to plot a cumulative event timeline. Default is ``True``.

        et_color : str, optional
            Specifies the color used for the secondary y-axis (event timeline axis), 
            including the ticks, labels, axis line, and the line representing 
            the cumulative number of events. Default is ``'orange'``.

        plot_cov : bool, optional
            If ``True`` and ``plot_vs='time'``, adds a secondary y-axis (``twiny``)
            to plot the coefficient of variation over time. Default is ``True``.

        cov_color : str, optional
            Specifies the color used for the secondary y-axis (event timeline axis), 
            including the ticks, labels, axis line, and the line representing 
            the coefficient og variation. Default is ``'blue'``.
            
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
            raises them to a power. Default is ``(1, 2)``.

        yscale : str, optional
            Specifies the scale of the y-axis. Common options include ``'linear'``,
            ``'log'``, ``'symlog'``, and ``'logit'``. Default is ``'log'``.

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
            Location of the size legend when ``size_legend`` is ``True``.
            Default is ``'upper right'``.

        ms_line : float, optional
            The magnitude threshold above which vertical lines will be added to 
            the plot. If ``None``, no vertical lines are added. Default is ``None``.

        ms_line_color : str, optional
            The color of the vertical lines. Accepts any Matplotlib-compatible 
            color string. Default is ``'red'``.

        ms_line_width : float, optional
            The thickness of the vertical lines. Default is 1.5.

        ms_line_style : str, optional
            The style of the vertical lines. Default is ``'-'``.

        ms_line_gradient : bool, optional
            If ``True``, the vertical lines will have a gradient effect, fading 
            from the pecified color to transparent along the y-axis. If ``False``, 
            the lines will be solid. Default is ``True``.                

        fig_size : tuple[float, float], optional
            Figure size for the plot. Default is ``(10, 5)``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is
            ``'interevent_time'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default
            is ``'jpg'``.

        Raises
        ------
        ValueError
            If an invalid configuration is provided.
        """
        if 'interevent_time' not in self._instance.data.columns:
            self.interevent_time(plot=False)

        rolling_mean = self._instance.data.interevent_time.rolling(
            window=window_size, min_periods=window_size
        ).mean()
        rolling_std = self._instance.data.interevent_time.rolling(
            window=window_size, min_periods=window_size
        ).std()

        rolling_cov = rolling_std / rolling_mean

        self._instance.data['cov'] = rolling_cov

        if plot:
            if kwargs.get('plot_cov', None) is False:
                raise ValueError(
                    "To plot just the interevent time, please use "
                    "`instance.interevent_time(plot=True)` instead."
                )
            else:
                self._plot_interevent_time(plot_cov=True, **kwargs)

    def fit_omori(
            self,
        ) -> dict:
        """
        Fit Omori's law to aftershock data.
        """
        raise NotImplementedError("fit_omori method is not yet implemented.")

    def _omori_law(
            self,
            t: ArrayLike,
            K: float,
            c: float,
            p: float
        ) -> ArrayLike:
        """
        Omori law model function.
        """
        return K / ((t + c) ** p)

    def _plot_interevent_time(
            self,
            plot_vs: str = 'time',
            plot_event_timeline: bool = True,
            et_color: str = 'red',
            plot_cov: bool = False,
            cov_color: str = 'blue',
            color_by: str = None,
            cmap: str = 'jet',
            size: float | str = 10,
            size_scale_factor: tuple[float, float] = (1, 2),
            yscale: str = 'log',
            color: str = 'grey',
            edgecolor: str = 'black',
            alpha: float = 0.75,
            size_legend: bool = False,
            size_legend_loc: str = 'upper right',
            ms_line: float = None,
            ms_line_color: str = 'orange',
            ms_line_width: float = 1.5,
            ms_line_style: str = '-',
            ms_line_gradient : bool = True,
            fig_size: tuple[float, float] = (10, 5),
            save_name: str = 'interevent_time',
            save_figure: bool = False,
            save_extension: str = 'jpg'
        ) -> None:
        """
        Plots inter-event times against any specified attribute, optionally 
        including the Coefficient of Variation (COV) and event timeline.
        """
        if plot_cov:
            if plot_vs != 'time':
                raise ValueError(
                    f"It is not possible to plot COV against '{plot_vs}'."
                    f" Please use `instance.interevent_time(plot_vs='{plot_vs}')` instead."
                )

        pu.set_style(styling.DEFAULT)
        fig, ax = plt.subplots(figsize=fig_size)

        plt_size = pu.process_size_parameter(size, self._instance.data, size_scale_factor)

        if color_by:
            fig.set_figwidth(
                fig_size[0] + (4 if plot_event_timeline and plot_cov else 2)
            )
            pu.plot_with_colorbar(
                ax=ax,
                data=self._instance.data,
                x=plot_vs,
                y='interevent_time',
                color_by=color_by,
                cmap=cmap,
                size=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                cbar_orientation='vertical',
                cbar_shrink=1,
                cbar_aspect=30,
                cbar_pad=(
                    0.16 if plot_event_timeline and plot_cov and plot_vs == 'time' else
                    0.11 if (plot_event_timeline or plot_cov) and plot_vs == 'time' else
                    0.03
                )
            )
        else:
            ax.scatter(
                x=self._instance.data[plot_vs],
                y=self._instance.data['interevent_time'],
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
                data=self._instance.data,
                size_scale_factor=size_scale_factor,
                alpha=alpha,
                size_legend_loc=size_legend_loc
            )

        if plot_vs == 'time':
            pu.format_x_axis_time(ax)
            pu.add_vertical_lines(
                ax=ax,
                data=self._instance.data,
                ms_line=ms_line,
                color=ms_line_color,
                linewidth=ms_line_width,
                linestyle=ms_line_style,
                gradient=ms_line_gradient
            )

            if plot_cov:
                pu.set_style(styling.CROSS_SECTION)

                ax_cov = ax.twinx()
                ax_cov.spines['right'].set_color(cov_color)
                ax_cov.tick_params(axis='y', colors=cov_color)
                ax_cov.set_ylabel('Coefficient of Variation (COV)', color=cov_color)

                ax_cov.plot(
                    self._instance.data[plot_vs],
                    self._instance.data['cov'],
                    color=cov_color,
                )
                try:
                    ax_cov.axhline(1, color=f'dark{cov_color}', ls='--')
                except:
                    ax_cov.axhline(1, color=f'black', ls='--')
                
                from matplotlib.transforms import blended_transform_factory 
                transform = blended_transform_factory(
                    ax_cov.transAxes, ax_cov.transData
                )

                ax_cov.text(
                    0.9, 1.0,
                    'COV = 1', 
                    transform=transform,
                    va='center',
                    ha='left',
                    bbox=dict(
                        facecolor='white',
                        edgecolor='black',
                        linewidth=0.75,
                        alpha=0.9,
                        pad=3
                    )
                )

            if plot_event_timeline:
                pu.set_style(styling.CROSS_SECTION)

                ax_et = ax.twinx()
                ax_et.spines['right'].set_color(et_color)
                ax_et.spines['right'].set_position(
                    ('axes', 1.1 if plot_cov else 1)
                )
                ax_et.set_ylabel(
                    'Cumulative no. of events', color=et_color
                )
                ax_et.tick_params(
                    axis='y', color=et_color, labelcolor=et_color
                )

                timeline = self._instance.data['time'].sort_values()
                cumulative_events = range(1, len(timeline) + 1)
                ax_et.plot(timeline, cumulative_events, color=et_color)
                ax_et.set_ylim(0)

        xlabel_map = {
            'time': None,
            'mag': 'Magnitude',
            'lat': 'Latitude [°]',
            'lon': 'Longitude [°]',
            'depth': 'Depth [Km]'
        }

        ax.set_yscale(yscale)
        ax.set_ylabel('Interevent Time [s]')
        ax.set_xlabel(xlabel_map.get(plot_vs, plot_vs))
        ax.grid(True, alpha=0.25, axis='x', linestyle=':')

        if save_figure:
            pu.save_figure(f'{save_name}{"_cov" if plot_cov else ""}', save_extension)

        pu.reset_style()
    
    @staticmethod
    def _disable_warnings() -> None:
        """
        Disables warnings.
        """
        warnings.simplefilter("ignore")
        warnings.filterwarnings("ignore")