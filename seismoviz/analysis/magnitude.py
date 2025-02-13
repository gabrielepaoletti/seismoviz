import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seismoviz.analysis.utils import styling
from seismoviz.analysis.utils import plot_utils as pu
from numpy.typing import ArrayLike


class Mc:
    """
    Class to compute the magnitude of completeness (Mc) using various methods.
    """

    def __init__(self, ma_instance: "MagnitudeAnalysis"):
        self._ma = ma_instance

    def maxc(self, bin_size: float, mags: ArrayLike = None) -> float:
        """
        Calculate the magnitude of completeness (Mc) using the MAXC method.
        """
        bins, events_per_bin, _ = self._ma.fmd(
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

    def gft(
            self,
            bin_size: float,
            mags: ArrayLike = None,
            gft_plot: bool = False,
            **kwargs
    ) -> float:
        """
        Estimate the magnitude of completeness (Mc) using the Goodness-of-Fit Test (GFT).
        """
        bins, _, cumulative_events = self._ma.fmd(
            bin_size=bin_size,
            plot=False,
            return_values=True,
            mags=mags
        )
        if not bins.size:
            return np.nan

        max_mc = self.maxc(bin_size=bin_size, mags=mags)
        results = []
        uncertainty_method = kwargs.get("uncertainty_method", "shi_bolt")

        for i, bin_mag in enumerate(bins):
            cutoff_magnitude = round(bin_mag, 1)
            result = self._ma._estimate_b_value(
                bin_size=bin_size,
                mc=cutoff_magnitude,
                mags=mags,
                plot=False,
                return_values=True,
                uncertainty_method=uncertainty_method
            )
            if result is None:
                continue

            _, a_value, b_value, _ = result

            synthetic_gr = 10 ** (a_value - b_value * bins)
            observed_count = cumulative_events[i:]
            synthetic_count = synthetic_gr[i:]
            total_observed = observed_count.sum()

            if total_observed == 0:
                continue

            R_value = (np.abs(observed_count - synthetic_count).sum() /
                       total_observed) * 100
            results.append((bin_mag, R_value))

        if not results:
            print("No valid results found, using MAXC estimate.")
            return max_mc

        _, R_values = zip(*results)
        for conf_level in [95, 90]:
            acceptable_R = 100 - conf_level
            for bin_mag, R_value in results:
                if R_value <= acceptable_R:
                    mc = round(bin_mag, 1)
                    break
            else:
                continue
            break
        else:
            mc = max_mc
            print("No fits within confidence levels, using MAXC estimate.")

        if gft_plot:
            self._plot_gft(
                mc=mc,
                bins=bins,
                R_values=R_values,
                bin_size=bin_size,
                mags=mags,
                **kwargs
            )
        return mc

    def mbs(
            self,
            bin_size: float,
            delta_magnitude: float = 0.4,
            min_completeness: float = -3,
            mags: ArrayLike = None,
            mbs_plot: bool = False,
            uncertainty_method: str = 'shi_bolt',
            **kwargs
    ) -> float:
        """
        Calculate the magnitude of completeness (Mc) using the Magnitude 
        Binning Stability (MBS) method.
        """
        bins, _, _ = self._ma.fmd(
            bin_size=bin_size,
            plot=False,
            return_values=True,
            mags=mags
        )
        if not bins.size:
            return np.nan

        maxc_completeness = self.maxc(bin_size=bin_size, mags=mags)
        results = []

        for bin_mag in bins:
            cutoff_magnitude = round(bin_mag, 1)
            result = self._ma._estimate_b_value(
                bin_size=bin_size,
                mc=cutoff_magnitude,
                mags=mags,
                plot=False,
                return_values=True,
                uncertainty_method=uncertainty_method
            )
            if result is None:
                continue

            _, _, b_value, unc_value = result
            results.append({
                'bin_mag': bin_mag,
                'b_value': b_value,
                'uncertainty': unc_value
            })

        if not results:
            print("No valid results found, using MAXC estimate.")
            return maxc_completeness

        bin_mags = np.array([res['bin_mag'] for res in results])
        b_values = np.array([res['b_value'] for res in results])
        uncertainty_values = np.array([res['uncertainty'] for res in results])

        number_of_bins_in_delta = int(round(delta_magnitude / bin_size))
        rolling_avg_b_values = np.full_like(b_values, np.nan)
        b_value_stability_checks = np.full_like(b_values, False, dtype=bool)

        for i in range(len(b_values)):
            end_idx = i + number_of_bins_in_delta + 1
            if end_idx > len(b_values):
                break

            window_b_values = b_values[i:end_idx]
            if np.any(np.isnan(window_b_values)):
                continue

            rolling_avg = np.mean(window_b_values)
            rolling_avg_b_values[i] = rolling_avg

            stability_check = np.abs(rolling_avg - b_values[i]) <= uncertainty_values[i]
            b_value_stability_checks[i] = stability_check

        if np.any(b_value_stability_checks):
            stable_bins = bin_mags[b_value_stability_checks]
            mc_candidates = stable_bins[stable_bins > min_completeness]
            mc = round(np.min(mc_candidates), 1) if mc_candidates.size > 0 else maxc_completeness
        else:
            mc = maxc_completeness

        if mbs_plot:
            self._plot_mbs(
                mc=mc,
                bins=bin_mags,
                individual_b_values=b_values,
                rolling_avg_b_values=rolling_avg_b_values,
                uncertainty_values=uncertainty_values,
                bin_size=bin_size,
                mags=mags,
                uncertainty_method=uncertainty_method,
                **kwargs
            )
        return mc

    def _count_decimals(self, number: float) -> int:
        """
        Calculate the number of decimal places in a given number.
        """
        decimal_str = str(number).split(".")[1] if "." in str(number) else ""
        return len(decimal_str)

    def _plot_gft(
            self,
            mc: float,
            bins: ArrayLike,
            R_values: ArrayLike,
            bin_size: float,
            mags: ArrayLike = None,
            save_figure: bool = False,
            save_name: str = 'gft',
            save_extension: str = 'jpg',
    ) -> None:
        """
        Plot the Goodness-of-Fit Test (GFT) results.
        """
        pu.set_style(styling.DEFAULT)
        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Goodness-of-Fit Test (GFT)")
        ax.scatter(
            bins, R_values,
            color='white',
            marker='o',
            edgecolor='black',
            linewidth=0.75,
            label="GFT R vs. $M_{c}$"
        )
        ax.plot(bins, R_values, color='black', linewidth=0.75)
        ax.axvline(mc, color='blue', linestyle='--', linewidth=1,
                   label=f"GFT $M_c$ = {round(mc, 1)}")
        ax.axhline(5, linestyle='--', color='grey', linewidth=1)
        ax.axhline(10, linestyle='--', color='grey', linewidth=1)
        ax.text(
            0.01, 0.42, "90% Conf.",
            color='grey',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='left'
        )
        ax.text(
            0.01, 0.22, "95% Conf.",
            color='grey',
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='bottom',
            horizontalalignment='left'
        )
        ax.set_ylim(0, 25)
        ax.set_xlabel("Cut-off magnitude", fontsize=12)
        ax.set_ylabel("GFT R statistic", fontsize=12)

        if mags is None:
            mags = self._ma._instance.data.mag

        min_tick_positions = np.arange(mags.min(), mags.max() + bin_size, bin_size)
        ax.set_xticks(min_tick_positions, minor=True)
        ax.grid(True, alpha=0.25, axis='x', linestyle=':')
        ax.legend(loc='best', frameon=False)

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()

    def _plot_mbs(
            self,
            mc: float,
            bins: ArrayLike,
            individual_b_values: ArrayLike,
            rolling_avg_b_values: ArrayLike,
            uncertainty_values: ArrayLike,
            bin_size: float,
            mags: ArrayLike = None,
            save_figure: bool = False,
            save_name: str = 'mbs',
            save_extension: str = 'jpg',
            uncertainty_method: str = 'shi_bolt',
    ) -> None:
        """
        Plot the Magnitude Binning Stability (MBS) results.
        """
        pu.set_style(styling.DEFAULT)

        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Magnitude Binning Stability (MBS)")
        ax.errorbar(
            bins, individual_b_values,
            yerr=uncertainty_values,
            color='black',
            fmt='o',
            capthick=1,
            capsize=3,
            label=f"$b-value$ at $M_{{c}}$ ({uncertainty_method})"
        )
        ax.scatter(
            bins, rolling_avg_b_values,
            color='white',
            marker='o',
            edgecolor='red',
            linewidth=0.75,
            label="Avg. $b-value$, $M_{c}$ to $M_{c} + \\Delta M$"
        )
        ax.axvline(
            mc, color='blue', linestyle='--', linewidth=1,
            label=f"MBS $M_c$ = {round(mc, 1)}"
        )
        ax.set_ylim(0, 4)
        ax.set_xlabel("Cut-off magnitude")
        ax.set_ylabel("$b-value$ estimate")

        if mags is None:
            mags = self._ma._instance.data.mag
        min_tick_positions = np.arange(mags.min(), mags.max() + bin_size, bin_size)
        ax.set_xticks(min_tick_positions, minor=True)
        ax.grid(True, alpha=0.25, axis='x', linestyle=':')
        ax.legend(loc='best', frameon=False)

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()


class Uncertainties:
    """
    Class to compute uncertainties for b-value estimations.
    """

    def __init__(self):
        pass

    def _shi_bolt(
            self,
            b_value: float,
            variance: float,
            num_events: int
    ) -> float:
        """
        Calculate the Shi & Bolt uncertainty for the b-value estimation.
        """
        return 2.3 * b_value**2 * np.sqrt(variance / num_events)

    def _aki(
            self,
            b_value: float,
            num_events: int
    ) -> float:
        """
        Calculate the Aki uncertainty for the b-value estimation.
        """
        return b_value / np.sqrt(num_events)

    def compute(
            self,
            method: str,
            b_value: float,
            num_events: int,
            variance: float = None
    ) -> float:
        """
        Compute the uncertainty for the b-value estimation based on the selected 
        method.
        """
        if method == "shi_bolt":
            if variance is None:
                raise ValueError(
                    "Variance must be provided for the ``'shi_bolt'`` method."
                )
            return self._shi_bolt(b_value, variance, num_events)
        elif method == "aki":
            return self._aki(b_value, num_events)
        else:
            raise ValueError(
                "Uncertainty method must be ``'shi_bolt'`` or ``'aki'``."
            )


class MagnitudeAnalysis:
    """
    Class for performing magnitude analysis on seismic event data.
    """

    def __init__(self, instance: object):
        self._instance = instance
        self.mc = Mc(self)
        self.uncertainties = Uncertainties()

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
            ms_line_gradient: bool = True,
            fig_size: tuple[float, float] = (10, 5),
            save_figure: bool = False,
            save_name: str = 'magnitude_time',
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plot seismic event magnitudes over time.

        Parameters
        ----------
        color_by : str, optional
            Column name to color the markers by (default is ``None``).

        cmap : str, optional
            The colormap to use if coloring by a variable (default is ``'jet'``).

        size : float or str, optional
            Marker size or a column name to scale marker sizes (default is ``10``).

        size_scale_factor : tuple of float, optional
            Factor to scale the marker sizes (default is ``(1, 2)``).

        color : str, optional
            Color for the markers if not coloring by a variable (default is ``'grey'``).

        edgecolor : str, optional
            Edge color for the markers (default is ``'black'``).

        alpha : float, optional
            Alpha transparency for the markers (default is ``0.75``).

        size_legend : bool, optional
            If ``True``, create a legend for the marker sizes (default is ``False``).

        size_legend_loc : str, optional
            Location for the size legend (default is ``'upper right'``).

        ms_line : float, optional
            Value to add vertical lines representing a magnitude scale (default is ``None``).

        ms_line_color : str, optional
            Color for the vertical lines (default is ``'red'``).

        ms_line_width : float, optional
            Line width for the vertical lines (default is ``1``).

        ms_line_style : str, optional
            Line style for the vertical lines (default is ``'-'``).

        ms_line_gradient : bool, optional
            If ``True``, apply a gradient to the vertical lines (default is ``True``).

        fig_size : tuple of float, optional
            Figure size for the plot. Default is ``(10, 5)``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if ``save_figure`` is ``True``. Default is ``'magnitude_time'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default is ``'jpg'``.
        """
        pu.set_style(styling.DEFAULT)

        fig, ax = plt.subplots(figsize=fig_size)
        ax.set_title("Magnitude-time distribution", fontweight="bold")

        plt_size = pu.process_size_parameter(
            size, self._instance.data, size_scale_factor
        )

        if color_by:
            fig.set_figwidth(fig_size[0] + 2)
            pu.plot_with_colorbar(
                ax=ax,
                data=self._instance.data,
                x="time",
                y="mag",
                color_by=color_by,
                cmap=cmap,
                size=plt_size,
                edgecolor=edgecolor,
                alpha=alpha,
                cbar_orientation="vertical",
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
        ax.set_ylabel("Magnitude")
        ax.set_ylim(self._instance.data.mag.min())
        ax.grid(True, alpha=0.25, axis="y", linestyle=":")

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()

    def fmd(
            self,
            bin_size: float,
            plot: bool = True,
            return_values: bool = False,
            mags: ArrayLike = None,
            **kwargs
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Calculate the frequency-magnitude distribution (FMD) for seismic events.

        Parameters
        ----------
        bin_size : float
            The size of the bin for the frequency-magnitude distribution.

        plot : bool, optional
            If ``True``, plot the FMD (default is ``True``).

        return_values : bool, optional
            If ``True``, return the computed values (default is ``False``).

        mags : ArrayLike, optional
            Array of magnitudes to use (default is ``None``).

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if ``save_figure`` is ``True``. Default is ``'fmd'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default is ``'jpg'``.

        Returns
        -------
        tuple of ArrayLike
            A tuple containing:
                - bins: The centers of the magnitude bins.
                - events_per_bin: The number of events per bin.
                - cumulative_events: The cumulative number of events.
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

        bin_edges = np.arange(lowest_bin - bin_size / 2,
                              highest_bin + bin_size,
                              bin_size)
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

    def b_value(
            self,
            bin_size: float,
            mc: str | float,
            uncertainty_method: str = 'shi_bolt',
            **kwargs
    ):
        """
        Estimate the b-value for seismic events and calculate the associated 
        uncertainty.

        Parameters
        ----------
        bin_size : float
            The size of the bin for the frequency-magnitude distribution.

        mc : str or float
            The method to estimate Mc (``'maxc'``, ``'gft'``, or ``'mbs'``) or a numerical 
            value for Mc.

        uncertainty_method : str, optional
            The method for uncertainty estimation (default is ``'shi_bolt'``).
        
        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if ``save_figure`` is ``True``. Default is ``'b-value'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default is ``'jpg'``.

        Returns
        -------
        tuple or None
            If mc is provided as a numerical value, returns the b-value estimation 
            tuple.
        """
        if isinstance(mc, str):
            mc = self._estimate_mc(bin_size, mc, plot_mc=True)
            return self._estimate_b_value(
                bin_size=bin_size,
                mc=mc,
                uncertainty_method=uncertainty_method,
                **kwargs
            )
        elif isinstance(mc, (int, float)):
            return self._estimate_b_value(
                bin_size=bin_size,
                mc=mc,
                uncertainty_method=uncertainty_method,
                **kwargs
            )
        else:
            raise ValueError("Mc value is not valid.")

    def b_value_over_time(
            self,
            bin_size: float,
            mc_method: str,
            window_type: str,
            window_size: int | str,
            step_size: int | str = None,
            uncertainty_method: str = 'shi_bolt',
            min_events_ratio: float = 0.5,
            plot: bool = True,
            return_values: bool = False,
            **kwargs
    ) -> dict | None:
        """
        Calculate the b-value over time windows.

        Parameters
        ----------
        bin_size : float
            The size of the bin for the frequency-magnitude distribution.

        mc_method : str
            The method to estimate the magnitude of completeness (``'maxc'``, ``'gft'``, 
            or ``'mbs'``).

        window_type : str
            The type of window to use (``'event'`` or ``'time'``).

        window_size : int or str
            The size of the window. Must be an ``int`` if window_type is ``'event'`` or 
            a ``str`` if ``'time'``.

        step_size : int or str, optional
            The step size for the window. Defaults to the window size if not 
            provided.

        uncertainty_method : str, optional
            The method for uncertainty estimation (default is ``'shi_bolt'``).

        min_events_ratio : float, optional
            The minimum ratio of events above Mc required (default is ``0.5``).

        plot : bool, optional
            If ``True``, plot the b-value over time (default is ``True``).

        return_values : bool, optional
            If ``True``, return the computed values as a dictionary (default is 
            ``False``).

        ms_line : float, optional
            Value to add vertical lines representing a magnitude scale (default is ``None``).

        ms_line_color : str, optional
            Color for the vertical lines (default is ``'red'``).

        ms_line_width : float, optional
            Line width for the vertical lines (default is ``1.5``).

        ms_line_style : str, optional
            Line style for the vertical lines (default is ``'-'``).

        ms_line_gradient : bool, optional
            If ``True``, apply a gradient to the vertical lines (default is ``True``).

        fig_size : tuple of float, optional
            Figure size for the plot. Default is ``(10, 5)``.

        save_figure : bool, optional
            If ``True``, saves the plot to a file. Default is ``False``.

        save_name : str, optional
            Base name for the file if ``save_figure`` is ``True``. Default is ``'b_value_over_time'``.

        save_extension : str, optional
            File format for the saved figure (e.g., ``'jpg'``, ``'png'``). Default is ``'jpg'``.

        Returns
        -------
        dict or None
            A dictionary with keys: ``times``, ``b_values``, ``mc_values``, ``uncertainties``,
            if ``return_values`` is ``True``; otherwise, ``None``.
        """
        if not (0 <= min_events_ratio <= 1):
            raise ValueError("min_events_ratio must be between 0 and 1.")

        data = self._instance.data.sort_values("time").reset_index(drop=True)
        times = []
        b_values = []
        mc_values = []
        uncertainty_values = []

        if window_type == "event":
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
                window_mags = window_data.mag.values
                window_mags = window_mags[~np.isnan(window_mags)]

                if len(window_mags) < 2:
                    continue

                mc_val = self._estimate_mc(
                    bin_size=bin_size,
                    method=mc_method,
                    plot_mc=False,
                    mags=window_mags
                )

                if np.isnan(mc_val):
                    continue

                threshold = mc_val - bin_size / 2
                events_above_mc = window_mags[window_mags >= threshold]

                if len(events_above_mc) < min_events_ratio * len(window_mags):
                    continue

                result = self._estimate_b_value(
                    bin_size=bin_size,
                    mc=mc_val,
                    mags=window_mags,
                    plot=False,
                    return_values=True,
                    uncertainty_method=uncertainty_method
                )

                if result is None:
                    continue

                _, _, b_val, unc_val = result

                if np.isnan(b_val):
                    continue

                middle_index = len(window_data) // 2

                times.append(window_data.time.iloc[middle_index])
                b_values.append(b_val)
                uncertainty_values.append(unc_val)
                mc_values.append(mc_val)

        elif window_type == "time":
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

            data.set_index("time", inplace=True)
            start_time = data.index.min()
            end_time = data.index.max()
            window_starts = pd.date_range(
                start=start_time, end=end_time, freq=step_size
            )

            for window_start in window_starts:
                window_end = window_start + pd.Timedelta(window_size)
                window_data = data.loc[window_start:window_end].reset_index()
                window_mags = window_data.mag.values
                window_mags = window_mags[~np.isnan(window_mags)]

                if len(window_mags) < 2:
                    continue

                mc_val = self._estimate_mc(
                    bin_size=bin_size,
                    method=mc_method,
                    plot_mc=False,
                    mags=window_mags
                )

                if np.isnan(mc_val):
                    continue

                threshold = mc_val - bin_size / 2
                events_above_mc = window_mags[window_mags >= threshold]

                if len(events_above_mc) < min_events_ratio * len(window_mags):
                    continue

                result = self._estimate_b_value(
                    bin_size=bin_size,
                    mc=mc_val,
                    mags=window_mags,
                    plot=False,
                    return_values=True,
                    uncertainty_method=uncertainty_method
                )

                if result is None:
                    continue

                _, _, b_val, unc_val = result

                if np.isnan(b_val):
                    continue

                times.append(window_start + (window_end - window_start) / 2)
                b_values.append(b_val)
                uncertainty_values.append(unc_val)
                mc_values.append(mc_val)

            data.reset_index(inplace=True)

        else:
            raise ValueError("window_type must be 'event' or 'time'.")

        if plot and times:
            self._plot_b_value_over_time(
                times=times,
                b_values=b_values,
                uncertainty_values=uncertainty_values,
                uncertainty_method=uncertainty_method,
                **kwargs
            )
        elif plot and not times:
            print("No data available to plot.")

        if return_values:
            return {
                "times": times,
                "b_values": b_values,
                "mc_values": mc_values,
                "uncertainties": uncertainty_values
            }

    def _estimate_mc(
            self,
            bin_size: float,
            method: str,
            plot_mc: bool = False,
            mags: ArrayLike = None
    ) -> float:
        """
        Estimate the catalog's magnitude of completeness (Mc) using the selected 
        method.
        """
        if method == "maxc":
            return self.mc.maxc(bin_size, mags=mags)
        elif method == "gft":
            return self.mc.gft(bin_size, mags=mags, gft_plot=plot_mc)
        elif method == "mbs":
            return self.mc.mbs(bin_size, mags=mags, mbs_plot=plot_mc)
        else:
            raise ValueError("Mc method is not valid.")

    def _estimate_b_value(
            self,
            bin_size: float,
            mc: float,
            mags: ArrayLike = None,
            plot: bool = True,
            return_values: bool = False,
            uncertainty_method: str = 'shi_bolt',
            **kwargs
    ) -> tuple[float, float, float, float] | None:
        """
        Estimate the b-value for seismic events and calculate the associated 
        uncertainty.
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
            uncertainty_value = np.nan
        else:
            mean_magnitude = np.mean(fm)
            delta_m = mean_magnitude - threshold
            if delta_m == 0:
                b_value = np.nan
                a_value = np.nan
                uncertainty_value = np.nan
            else:
                b_value = log10_e / delta_m
                a_value = np.log10(num_events) + b_value * mag_compl
                variance = np.var(fm, ddof=1)
                uncertainty_value = self.uncertainties.compute(
                    uncertainty_method, b_value, num_events, variance
                )
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
                uncertainty_value=uncertainty_value,
                uncertainty_method=uncertainty_method,
                **kwargs
            )
        if return_values:
            return (
                mag_compl,
                a_value,
                b_value,
                uncertainty_value
            )
        
    def _count_decimals(self, number: float) -> int:
        """
        Calculate the number of decimal places in a given number.
        """
        decimal_str = str(number).split(".")[1] if "." in str(number) else ""
        return len(decimal_str)

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
        Plot the frequency-magnitude distribution (FMD).
        """
        pu.set_style(styling.DEFAULT)
        
        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Frequency-magnitude distribution", fontsize=14, fontweight="bold")
        ax.scatter(
            bins, cumulative_events,
            color="white",
            marker="o",
            edgecolor="black",
            linewidth=0.75,
            label="Cumulative no. of events"
        )
        ax.scatter(
            bins, events_per_bin,
            color="white",
            marker="o",
            edgecolor="red",
            linewidth=0.75,
            label="No. of events per mag. bin"
        )
        
        ax.set_yscale("log")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Frequency")
        ax.legend(loc="best", frameon=False)
        
        min_tick_positions = np.arange(min(bins), max(bins) + bin_size, bin_size)
        ax.set_xticks(min_tick_positions, minor=True)
        ax.grid(True, alpha=0.25, axis="x", linestyle=":")

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
            uncertainty_value: float,
            uncertainty_method: str = 'shi_bolt',
            save_figure: bool = False,
            save_name: str = 'b-value',
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plot the b-value estimation with the frequency-magnitude distribution.
        """
        pu.set_style(styling.DEFAULT)
        
        _, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("b-value", fontsize=14, fontweight="bold")
        
        below_mc = bins < mc
        above_mc = bins >= mc
        
        ax.scatter(
            bins[above_mc],
            cumulative_events[above_mc],
            color="white",
            marker="o",
            edgecolor="black",
            linewidth=0.75,
            label="Cumulative no. of events"
        )
        ax.scatter(
            bins[below_mc],
            cumulative_events[below_mc],
            color="black",
            marker="x",
            linewidth=0.75
        )
        ax.scatter(
            bins[above_mc],
            events_per_bin[above_mc],
            color="white",
            marker="o",
            edgecolor="red",
            linewidth=0.75,
            label="No. of events per mag. bin"
        )
        ax.scatter(
            bins[below_mc],
            events_per_bin[below_mc],
            color="red",
            marker="x",
            linewidth=0.75
        )
        ax.plot(
            bins[above_mc],
            10 ** (a_value - b_value * bins[above_mc]),
            color="blue"
        )
        ax.axvline(x=mc, color="gray", linestyle="--", linewidth=1)
        mc_index = np.where(bins == mc)
        ax.scatter(
            bins[mc_index],
            cumulative_events[mc_index],
            color="black",
            marker="o",
            s=50
        )
        ax.scatter(
            bins[mc_index],
            events_per_bin[mc_index],
            color="red",
            marker="o",
            s=50
        )
        ax.set_yscale("log")
        ax.set_xlabel("Magnitude")
        ax.set_ylabel("Frequency")
        ax.set_ylim(1, 10 ** np.ceil(np.log10(len(self._instance.data.mag))))
        ax.legend(loc="upper right", frameon=False)
        text_str = (f"$M_c$ = {mc}\n"
                    f"$b-value$ = {round(b_value, 3)} ± {round(uncertainty_value, 3)}"
                    f" ({uncertainty_method})")
        ax.text(
            0.017, 0.03, text_str,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(facecolor="white", edgecolor="black")
        )
        min_tick_positions = np.arange(min(bins), max(bins) + bin_size, bin_size)
        ax.set_xticks(min_tick_positions, minor=True)
        ax.grid(True, alpha=0.25, axis="x", linestyle=":")

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()

    def _plot_b_value_over_time(
            self,
            times: ArrayLike,
            b_values: ArrayLike,
            uncertainty_values: ArrayLike,
            uncertainty_method: str = 'shi_bolt',
            ms_line: float = None,
            ms_line_color: str = 'red',
            ms_line_width: float = 1.5,
            ms_line_style: str = '-',
            ms_line_gradient: bool = True,
            fig_size: tuple[float, float] = (10, 5),
            save_figure: bool = False,
            save_name: str = 'b_value_over_time',
            save_extension: str = 'jpg'
    ) -> None:
        """
        Plot the b-value over time with associated uncertainty.
        """
        pu.set_style(styling.DEFAULT)

        _, ax = plt.subplots(figsize=fig_size)
        ax.set_title("b-value over time", fontweight="bold")
        ax.plot(times, b_values, color="black", lw=0.75, label="b-value")
        
        pu.add_vertical_lines(
            ax=ax,
            data=self._instance.data,
            ms_line=ms_line,
            color=ms_line_color,
            linewidth=ms_line_width,
            linestyle=ms_line_style,
            gradient=ms_line_gradient
        )
        
        lower = np.array(b_values) - np.array(uncertainty_values)
        upper = np.array(b_values) + np.array(uncertainty_values)
        ax.fill_between(
            times, lower, upper,
            color="gray", alpha=0.3,
            label=f"Uncertainty ({uncertainty_method})"
        )
        ax.set_ylabel("$b-value$")
        ax.set_ylim(min(b_values), max(b_values) * 1.1)
        ax.grid(True, axis="x", alpha=0.25, linestyle=":")
        ax.legend(loc="best", frameon=False)
        pu.format_x_axis_time(ax)

        if save_figure:
            pu.save_figure(save_name, save_extension)

        plt.show()
        pu.reset_style()