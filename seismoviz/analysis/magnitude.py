import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seismoviz.analysis.utils import styling
from seismoviz.analysis.utils import plot_utils as pu

from numpy.typing import ArrayLike


class MagnitudeAnalysis:
    def __init__(self, instance: object):
        self._instance = instance

    def magnitude_time(
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
            ms_line: float = None,
            ms_line_color: str = 'red',
            ms_line_width: float = 1,
            ms_line_style: str = '-',
            ms_line_gradient : bool = True,
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

        hl_ms : float
            The minimum magnitude threshold for highlighting events. Vertical 
            lines will be added to the plot at dates corresponding to seismic 
            events with a magnitude greater than or equal to this value.

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
            ``'magnitude_time'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default
            is ``'jpg'``.

        Returns
        -------
        None
            A magnitude-time plot.
        """
        pu.set_style(styling.DEFAULT)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title('Magnitude-time distribution', fontweight='bold')

        plt_size = pu.process_size_parameter(
            size, self._instance.data, size_scale_factor
        )

        if color_by:
            fig.set_figwidth(fig_size[0] + 2)
            pu.plot_with_colorbar(
                ax=ax,
                data=self._instance.data,
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
                x=self._instance.data.time,
                y=self._instance.data.mag,
                c=color,
                s=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                lw=0.25
            )

        pu.add_vertical_lines(
            ax=ax,
            data=self._instance.data,
            ms_line=ms_line,
            color=ms_line_color,
            linewidth=ms_line_width,
            linestyle=ms_line_style,
            gradient=ms_line_gradient
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

        pu.format_x_axis_time(ax)
        ax.set_ylabel('Magnitude')
        ax.set_ylim(self._instance.data.mag.min())
        ax.grid(True, alpha=0.25, axis='y', linestyle=':')

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()


    def fmd(
            self,
            bin_size: float,
            plot: bool = True,
            return_values: bool = False,
            mags: np.ndarray = None,
            **kwargs
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
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

        tuple[ArrayLike, ArrayLike, ArrayLike]
            - ``bins`` : ArrayLike
                Array of magnitude bin edges.
            - ``events_per_bin`` : ArrayLike
                Array with the number of events in each magnitude bin.
            - ``cumulative_events`` : ArrayLike
                Array with the cumulative number of events for magnitudes greater than 
                or equal to each bin.
        """
        if mags is None:
            mags = self._instance.data.mag.values

        mags = mags[~np.isnan(mags)]

        if len(mags) == 0:
            if return_values:
                return np.array([]), np.array([]), np.array([])
            else:
                print("No valid magnitude data available to compute FMD.")
                return

        lowest_bin = np.floor(np.min(mags) / bin_size) * bin_size
        highest_bin = np.ceil(np.max(mags) / bin_size) * bin_size

        if lowest_bin >= highest_bin:
            if return_values:
                return np.array([]), np.array([]), np.array([])
            else:
                print("Invalid magnitude range to compute FMD.")
                return

        bin_edges = np.arange(lowest_bin - bin_size / 2, highest_bin + bin_size, bin_size)

        events_per_bin, _ = np.histogram(mags, bins=bin_edges)
        cumulative_events = np.cumsum(events_per_bin[::-1])[::-1]
        bins = bin_edges[:-1] + bin_size / 2

        if plot:
            self._plot_fmd(
                bins=bins,
                events_per_bin=events_per_bin,
                cumulative_events=cumulative_events,
                bin_size=bin_size,
                **kwargs
            )

        if return_values:
            return bins, events_per_bin, cumulative_events

    def b_value(self, bin_size: float, mc: str | float, **kwargs):
        """
        Estimates the b-value for seismic events, and calculates the associated 
        uncertainties.

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

        return_values : bool, optional
            If ``True``, returns the calculated values. Default is ``False``.            

        plot_uncertainty : str, optional
            Type of uncertainty to display in the plot. Options are ``'shi_bolt'`` 
            for Shi and Bolt uncertainty and ``'aki'`` for Aki uncertainty. 
            Default is ``'shi_bolt'``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is 
            ``'b-value'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). 
            Default is ``'jpg'``.

        Returns
        -------
        .. warning::
            Values are returned only if ``return_values`` argument is set to 
            ``True``
            
        tuple[float, float, float, float]
            - ``mc``: float
                The magnitude of completeness value.
            - ``a_value`` : float
                The a-value, representing the logarithmic scale of the seismicity 
                rate.
            - ``b_value`` : float
                The b-value, indicating the relative occurrence of large and 
                small earthquakes.
            - ``aki_uncertainty`` : float
                The Aki uncertainty in the b-value estimation.
            - ``shi_bolt_uncertainty`` : float
                The Shi and Bolt uncertainty in the b-value estimation.
        
        Raise
        -----
        ValueError
            If the selected Mc type or value is not valid.
        """
        if isinstance(mc, str):
            mc = self._estimate_mc(bin_size, mc)
            return self._estimate_b_value(
                bin_size=bin_size, mc=mc, **kwargs
            )
        elif isinstance(mc, int) or isinstance(mc, float):
            return self._estimate_b_value(
                bin_size=bin_size, mc=mc, **kwargs
            )
        else:
            raise ValueError('Mc value is not valid.')

    def b_value_over_time(
        self,
        bin_size: float,
        mc_method: str,
        window_type: str,
        window_size: int | str,
        step_size: int | str = None,
        uncertainty: str = 'shi_bolt',
        min_events_ratio: float = 0.5,
        plot: bool = True,
        return_values: bool = False,
        **kwargs
    ) -> tuple[list, list, list, list, list] | None:
        """
        Calculates the b-value over time windows, either by grouping a fixed 
        number of events or by fixed time intervals.

        Parameters
        ----------
        bin_size : float
            The size of each magnitude bin for calculating frequency-magnitude 
            distribution.

        mc_method : str, optional
            The method to calculate the magnitude of completeness.

        window_type : str, optional
            The type of windowing method to use: ``'event'`` for a fixed number 
            of events per window, or ``'time'`` for time-based windows.

        window_size : int or str, optional
            The size of each window. For ``'event'`` windowing, this should be 
            an integer specifying the number of events. For ``'time'`` windowing, 
            this should be a string representing a pandas time offset alias (e.g., 
            ``'1D'`` for one day).

        step_size : int or str, optional
            The step size for moving the window. For ``'event'`` windowing, this 
            should be an integer. For ``'time'`` windowing, this should be a string 
            representing a pandas time offset alias. If not provided, defaults 
            to the window_size (non-overlapping windows).

        uncertainty : str, optional
            Type of uncertainty to display in the plot. Options are ``'shi_bolt'`` 
            for Shi & Bolt uncertainty and ``'aki'`` for Aki uncertainty. Default 
            is ``'shi_bolt'``.

        min_events_ratio : float, optional
            The minimum fraction (between 0 and 1) of events above the magnitude 
            of completeness required to calculate the b-value for a window. Default 
            is 0.5 (50%).

        plot : bool, optional
            If ``True``, plots the frequency-magnitude distribution with the 
            calculated b-value curve. Default is ``True``.

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

        return_values : bool, optional
            If ``True``, returns the calculated values. Default is ``False``.            

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if `save_figure` is ``True``. Default is 
            ``'b-value'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). 
            Default is ``'jpg'``.

        Returns
        -------
        .. warning::
            Values are returned only if ``return_values`` argument is set to 
            ``True``

        tuple[list, list, list, list]
                - ``times`` : list
                    List of times corresponding to each b-value calculation.
                - ``b_values`` : list
                    List of calculated b-values.
                - ``aki_uncs`` : list
                    List of Aki uncertainties associated with each b-value.
                - ``shi_bolt_uncs`` : list
                    List of Shi and Bolt uncertainties associated with each b-value.

        Raises
        ------
        ValueError
            If invalid values or types are provided for ``window_type``, 
            ``window_size``, or ``step_size``, or if ``min_events_ratio`` is 
            not between 0 and 1.
        """
        if not (0 <= min_events_ratio <= 1):
                raise ValueError("min_events_ratio must be between 0 and 1.")

        data = self._instance.data.sort_values('time').reset_index(drop=True)

        mc_values = []
        b_values, times = [], []
        aki_uncs, shi_bolt_uncs = [], []

        if window_type == 'event':
            if not isinstance(window_size, int):
                raise ValueError(
                    "window_size must be an integer when window_type is 'event'."
                )

            if step_size is None:
                step_size = window_size

            if not isinstance(step_size, int):
                raise ValueError(
                    "step_size must be an integer when window_type is 'event'."
                )

            total_events = data.shape[0]
            indices = range(0, total_events - window_size + 1, step_size)

            for idx in indices:
                window_data = data.iloc[idx:idx + window_size]
                mags = window_data.mag.values
                mags = mags[~np.isnan(mags)]

                if len(mags) < 2:
                    continue

                mc = self._estimate_mc(bin_size=bin_size, method=mc_method, mags=mags)

                if np.isnan(mc):
                    continue

                # Calculate the number of events above Mc
                threshold = mc - bin_size / 2
                events_above_mc = mags[mags >= threshold]

                # Check if the fraction of events above Mc meets the minimum requirement
                if len(events_above_mc) < min_events_ratio * len(mags):
                    continue  # Skip this window

                result = self._estimate_b_value(
                    bin_size=bin_size,
                    mc=mc,
                    mags=mags,
                    plot=False,
                    return_values=True
                )

                if result is None:
                    continue

                _, _, b_value, aki_uncertainty, shi_bolt_uncertainty = result

                if np.isnan(b_value):
                    continue

                middle_index = len(window_data) // 2
                time = window_data.time.iloc[middle_index]
                times.append(time)
                b_values.append(b_value)
                aki_uncs.append(aki_uncertainty)
                shi_bolt_uncs.append(shi_bolt_uncertainty)
                mc_values.append(mc)

        elif window_type == 'time':
            if not isinstance(window_size, str):
                raise ValueError(
                    "window_size must be a string when window_type is 'time'."
                )

            if step_size is None:
                step_size = window_size

            if not isinstance(step_size, str):
                raise ValueError(
                    "step_size must be a string when window_type is 'time'."
                )

            data.set_index('time', inplace=True)
            start_time = data.index.min()
            end_time = data.index.max()
            window_starts = pd.date_range(start=start_time, end=end_time, freq=step_size)

            for window_start in window_starts:
                window_end = window_start + pd.Timedelta(window_size)
                window_data = data.loc[window_start:window_end].reset_index()

                mags = window_data.mag.values
                mags = mags[~np.isnan(mags)]

                if len(mags) < 2:
                    continue

                mc = self._estimate_mc(bin_size=bin_size, method=mc_method, mags=mags)

                if np.isnan(mc):
                    continue

                threshold = mc - bin_size / 2
                events_above_mc = mags[mags >= threshold]

                if len(events_above_mc) < min_events_ratio * len(mags):
                    continue

                result = self._estimate_b_value(
                    bin_size=bin_size,
                    mc=mc,
                    mags=mags,
                    plot=False,
                    return_values=True
                )

                if result is None:
                    continue

                _, _, b_value, aki_uncertainty, shi_bolt_uncertainty = result

                if np.isnan(b_value):
                    continue

                times.append(window_start + (window_end - window_start) / 2)
                b_values.append(b_value)
                aki_uncs.append(aki_uncertainty)
                shi_bolt_uncs.append(shi_bolt_uncertainty)
                mc_values.append(mc)

            data.reset_index(inplace=True)

        else:
            raise ValueError("window_type must be 'event' or 'time'.")

        if plot and times:
            selected_uncertainties = (
                aki_uncs if uncertainty == 'aki' else shi_bolt_uncs
            )
            uncertainty_label = 'Aki' if uncertainty == 'aki' else 'Shi & Bolt'

            self._plot_b_value_over_time(
                times=times,
                b_values=b_values,
                uncertainties=selected_uncertainties,
                uncertainty_label=uncertainty_label,
                **kwargs
            )
        elif plot and not times:
            print("No data available to plot.")

        if return_values:
            return times, b_values, aki_uncs, shi_bolt_uncs, mc_values

    def _maxc(self, bin_size: float, mags: np.ndarray = None) -> float:
        """
        Calculates the magnitude of completeness (Mc) for the seismic catalog
        using the MAXC method.
        """
        bins, events_per_bin, _ = self.fmd(
            bin_size=bin_size,
            plot=False,
            return_values=True,
            mags=mags
        )

        if len(events_per_bin) == 0 or len(bins) == 0:
            return np.nan

        max_event_count_bin = bins[np.argmax(events_per_bin)]
        decimals = self._count_decimals(bin_size)
        return round(max_event_count_bin, decimals)
    
    def _estimate_mc(
            self,
            bin_size: float,
            method: str,
            mags: np.ndarray = None
    ) -> float:
        """
        Estimates catalog's magnitude of completeness (Mc) using the selected
        method.
        """
        if method == 'maxc':
            return self._maxc(bin_size, mags=mags)
        else:
            raise ValueError('Mc value is not valid.')

    def _estimate_b_value(
            self,
            bin_size: float,
            mc: float,
            mags: np.ndarray = None,
            plot: bool = True,
            return_values: bool = False,
            **kwargs
    ) -> tuple[float, float, float, float]:
        """
        Estimates the b-value for seismic events, and calculates the associated 
        uncertainties.
        """
        decimals = self._count_decimals(bin_size)

        mag_compl = round(mc, decimals)
        threshold = mag_compl - (bin_size / 2)
        log10_e = np.log10(np.exp(1))

        if mags is None:
            mags = self._instance.data.mag.values

        mags = mags[~np.isnan(mags)]
        fm = mags[mags >= threshold]
        num_events = fm.size

        if num_events < 2 or np.std(fm) == 0:
            a_value, b_value = np.nan, np.nan
            shi_bolt_uncertainty, aki_uncertainty = np.nan, np.nan
        else:
            mean_magnitude = np.mean(fm)
            delta_m = mean_magnitude - threshold

            if delta_m == 0:
                b_value = np.nan
                aki_uncertainty = np.nan
                shi_bolt_uncertainty = np.nan
            else:
                b_value = log10_e / delta_m
                a_value = np.log10(num_events) + b_value * mag_compl
                aki_uncertainty = b_value / np.sqrt(num_events)
                variance = np.var(fm, ddof=1)
                shi_bolt_uncertainty = 2.3 * b_value**2 * np.sqrt(variance / num_events)

        if plot:
            bins, events_per_bin, cumulative_events = self.fmd(
                bin_size=bin_size,
                plot=False,
                return_values=True,
                mags=mags
            )
            bins = np.round(bins, decimals)

            self._plot_b_value(
                bins=bins,
                events_per_bin=events_per_bin,
                cumulative_events=cumulative_events,
                bin_size=bin_size,
                mc=mag_compl,
                a_value=a_value,
                b_value=b_value,
                shi_bolt_uncertainty=shi_bolt_uncertainty,
                aki_uncertainty=aki_uncertainty,
                **kwargs
            )

        if return_values:
            return mag_compl, a_value, b_value, aki_uncertainty, shi_bolt_uncertainty

    def _plot_fmd(
            self,
            bins: ArrayLike, 
            events_per_bin: ArrayLike, 
            cumulative_events: ArrayLike, 
            bin_size: float, 
            save_figure: bool = False, 
            save_name: str = 'fmd', 
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plots the frequency-magnitude distribution (FMD).
        """
        pu.set_style(styling.DEFAULT)
        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(
            'Frequency-magnitude distribution', fontsize=14, fontweight='bold'
        )

        ax.scatter(
            bins, cumulative_events, color='white', marker='o', 
            edgecolor='black', linewidth=0.75, label='Cumulative no. of events'
        )
        ax.scatter(
            bins, events_per_bin, color='white', marker='o', 
            edgecolor='red', linewidth=0.75, label='No. of events per mag. bin'
        )

        ax.set_yscale('log')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Frequency')
        ax.legend(loc='best', frameon=False)

        min_tick_positions = np.arange(min(bins), max(bins) + bin_size, bin_size)
        ax.set_xticks(min_tick_positions, minor=True)

        ax.grid(True, alpha=0.25, axis='x', linestyle=':')

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()

    def _plot_b_value(
            self,
            bins: ArrayLike,
            events_per_bin: ArrayLike,
            cumulative_events: ArrayLike,
            bin_size: float,
            mc: float,
            a_value: float,
            b_value: float,
            shi_bolt_uncertainty: float,
            aki_uncertainty: float,
            plot_uncertainty: str = 'shi_bolt',
            save_figure: bool = False,
            save_name: str = 'b-value',
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plots the b-value estimation with the FMD.
        """
        pu.set_style(styling.DEFAULT)

        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_title('b-value', fontsize=14, fontweight='bold')

        below_mc = bins < mc
        above_mc = bins >= mc

        ax.scatter(
            bins[above_mc], cumulative_events[above_mc], color='white', marker='o', 
            edgecolor='black', linewidth=0.75, label='Cumulative no. of events'
        )
        ax.scatter(
            bins[below_mc], cumulative_events[below_mc], color='black', marker='x', 
            linewidth=0.75
        )

        ax.scatter(
            bins[above_mc], events_per_bin[above_mc], color='white', marker='o', 
            edgecolor='red', linewidth=0.75, label='No. of events per mag. bin'
        )
        ax.scatter(
            bins[below_mc], events_per_bin[below_mc], color='red', marker='x', 
            linewidth=0.75
        )

        ax.plot(
            bins[above_mc], (10**(a_value - (b_value * bins[above_mc]))), color='blue'
        )

        ax.axvline(x=mc, color='gray', linestyle='--', linewidth=1)

        mc_index = np.where(bins == mc)
        ax.scatter(
            bins[mc_index], cumulative_events[mc_index], color='black', marker='o', 
            s=50
        )
        ax.scatter(
            bins[mc_index], events_per_bin[mc_index], color='red', marker='o', 
            s=50
        )

        ax.set_yscale('log')
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Frequency')
        ax.set_ylim(1, 10**np.ceil(np.log10(len(self._instance.data.mag))))

        ax.legend(loc='upper right', frameon=False)

        if plot_uncertainty == 'shi_bolt':
            plot_uncert = shi_bolt_uncertainty
        elif plot_uncertainty == 'aki':
            plot_uncert = aki_uncertainty
        else:
            raise ValueError("Uncertainty must be 'shi_bolt' or 'aki'.")

        text_str = (
            f'$M_c$ = {mc}\n'
            f'$b-value$ = {round(b_value, 3)} ± {round(plot_uncert, 3)}'
        )
        ax.text(
            0.017, 0.03, text_str, transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='white', edgecolor='black')
        )

        min_tick_positions = np.arange(
            min(bins), max(bins) + bin_size, bin_size
        )
        ax.set_xticks(min_tick_positions, minor=True)

        ax.grid(True, alpha=0.25, axis='x', linestyle=':')

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()

    def _plot_b_value_over_time(
            self,
            times: ArrayLike,
            b_values: ArrayLike,
            uncertainties: ArrayLike,
            uncertainty_label: str,
            ms_line: float = None,
            ms_line_color: str = 'red',
            ms_line_width: float = 1.5,
            ms_line_style: str = '-',
            ms_line_gradient : bool = True,
            save_figure: bool = False,
            save_name: str = 'b_value_over_time',
            save_extension: str = 'jpg',
            fig_size: tuple[float, float] = (10, 5),
    ) -> None:
        """
        Plots the b-value over time with associated uncertainties.
        """
        pu.set_style(styling.DEFAULT)

        _, ax = plt.subplots(figsize=fig_size)
        ax.set_title('b-value over time', fontweight='bold')

        ax.plot(times, b_values, color='black', lw=0.75, label='b-value')
        pu.add_vertical_lines(
            ax=ax,
            data=self._instance.data,
            ms_line=ms_line,
            color=ms_line_color,
            linewidth=ms_line_width,
            linestyle=ms_line_style,
            gradient=ms_line_gradient
        )

        lower = np.array(b_values) - np.array(uncertainties)
        upper = np.array(b_values) + np.array(uncertainties)
        ax.fill_between(
            times, lower, upper, color='gray', alpha=0.3,
            label=f'Uncertainty ({uncertainty_label})'
        )

        ax.set_ylabel('$b-value$')
        ax.set_ylim(min(b_values))
        ax.grid(True, axis='x', alpha=0.25, linestyle=':')
        ax.legend(loc='best', frameon=False)

        pu.format_x_axis_time(ax)

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()

    def _count_decimals(self, number):
        """
        Calculate the number of decimal places in a given number.
        """
        decimal_str = str(number).split(".")[1] if "." in str(number) else ""
        return len(decimal_str)