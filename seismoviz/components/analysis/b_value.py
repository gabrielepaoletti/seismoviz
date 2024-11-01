import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from seismoviz.components.common import styling
from seismoviz.components.common.base_plotter import BasePlotter


class BValueCalculator:
    def __init__(self, catalog: type):
        self.ct = catalog
        self.bp = BasePlotter()
        
    def fmd(
        self,
        bin_size: float,
        plot: bool = False,
        save_figure: bool = False,
        save_name: str = 'fmd',
        save_extension: str = 'jpg'
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates the frequency-magnitude distribution (FMD) for seismic events, 
        which represents the number of events in each magnitude bin.

        Parameters
        ----------
        bin_size : float
            The size of each magnitude bin.

        plot : bool, optional
            If True, plots the FMD. Default is False.

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
        lowest_bin = np.floor(np.min(self.ct.data.mag) / bin_size) * bin_size
        highest_bin = np.ceil(np.max(self.ct.data.mag) / bin_size) * bin_size
        bin_edges = np.arange(lowest_bin - bin_size / 2, highest_bin + bin_size, bin_size)

        events_per_bin, _ = np.histogram(self.ct.data.mag, bins=bin_edges)
        cumulative_events = np.cumsum(events_per_bin[::-1])[::-1]
        bins = bin_edges[:-1] + bin_size / 2

        if plot:
            self.bp.set_style(styling.DEFAULT)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title('Frequency-magnitude distribution', fontsize=14, fontweight='bold')

            ax.scatter(
                bins, cumulative_events, color='white', marker='o', 
                edgecolor='black', linewidth=0.75, label='Cumulative no. of events'
            )
            ax.scatter(
                bins, events_per_bin, color='white', marker='o', 
                edgecolor='red', linewidth=0.75, label='No. of events per mag. bin'
            )

            ax.set_yscale('log')
            ax.set_xlabel('Magnitude', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.legend(loc='best', frameon=False, fontsize=12)

            min_tick_positions = np.arange(min(bins), max(bins) + bin_size, bin_size)
            ax.set_xticks(min_tick_positions, minor=True)

            ax.grid(True, alpha=0.25, axis='x', linestyle=':')

            if save_figure:
                self.save_figure(fig, save_name, save_extension)

            plt.show()
            self.bp.reset_style()

        return bins, events_per_bin, cumulative_events

    def estimate_b_value(
            self,
            bin_size: float,
            mc: str | float,
            plot: bool = True,
            plot_uncertainty: str = 'shi_bolt',
            save_figure: bool = False,
            save_name: str = 'b-value',
            save_extension: str = 'jpg'
    ) -> tuple[float, float, float, float]:
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
        threshold = round(mc, 1) - bin_size / 2
        log10_e = np.log10(np.exp(1))

        fm = self.ct.data.mag[self.ct.data.mag > threshold].values
        num_events = fm.size

        if num_events < 2:
            a_value, b_value = np.nan, np.nan
            shi_bolt_uncertainty, aki_uncertainty = np.nan, np.nan
        else:
            mean_magnitude = np.mean(fm)
            delta_m = mean_magnitude - threshold
            b_value = log10_e / delta_m
            a_value = np.log10(num_events) + b_value * mc
            aki_uncertainty = b_value / np.sqrt(num_events)
            variance = np.var(fm, ddof=1)
            shi_bolt_uncertainty = 2.3 * b_value**2 * np.sqrt(variance / num_events)

        if plot:
            self.bp.set_style(styling.DEFAULT)
            bins, events_per_bin, cumulative_events = self.fmd(bin_size=bin_size)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title('b-value', fontsize=14, fontweight='bold')

            ax.scatter(
                bins, cumulative_events, color='white', marker='o', edgecolor='black',
                linewidth=0.75, label='Cumulative no. of events'
            )
            ax.scatter(
                bins, events_per_bin, color='white', marker='o', edgecolor='red',
                linewidth=0.75, label='No. of events per mag. bin'
            )
            ax.plot(bins, (10**(a_value - (b_value * bins))), color='blue')

            ax.set_yscale('log')
            ax.set_xlabel('Magnitude')
            ax.set_ylabel('Frequency')
            ax.set_ylim(1, 10**np.ceil(np.log10(self.ct.data.shape[0])))

            ax.legend(loc='upper right', frameon=False)

            if plot_uncertainty == 'shi_bolt':
                plot_uncert = shi_bolt_uncertainty
            elif plot_uncertainty == 'aki':
                plot_uncert = aki_uncertainty
            else:
                raise ValueError("Uncertainty must be 'shi_bolt' or 'aki'.")

            text_str = (
                f'$M_c$ = {round(mc, 1)}\n'
                f'$b-value$ = {round(b_value, 3)} ± {round(plot_uncert, 3)}'
            )
            ax.text(
                0.017, 0.03, text_str, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='black')
            )

            min_tick_positions = np.arange(min(bins), max(bins) + bin_size, bin_size)
            ax.set_xticks(min_tick_positions, minor=True)

            ax.grid(True, alpha=0.25, axis='x', linestyle=':')

            if save_figure:
                self.save_figure(fig, save_name, save_extension)

            plt.show()
            self.bp.reset_style()

        return a_value, b_value, aki_uncertainty, shi_bolt_uncertainty

    def _maxc(self, bin_size: float) -> float:
        """
        Calculates the magnitude of completeness (Mc) for the seismic catalog. 

        Parameters
        ----------
        bin_size : float
            The size of each magnitude bin for calculating the frequency-magnitude 
            distribution (FMD).

        Returns
        -------
        float
            The magnitude of completeness (Mc), rounded to the nearest 0.1.
        """
        bins, events_per_bin, _ = self.fmd(bin_size)
        max_event_count_bin = bins[np.argmax(events_per_bin)]
        mc = round(max_event_count_bin, 1)
        
        return mc