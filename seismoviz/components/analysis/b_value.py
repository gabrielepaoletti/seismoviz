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
        lowest_bin = np.floor(min(self.ct.data.mag) / bin_size) * bin_size
        highest_bin = np.ceil(max(self.ct.data.mag) / bin_size) * bin_size

        bins = np.arange(lowest_bin, highest_bin + bin_size, bin_size)
        cumulative_events = np.zeros(len(bins))

        for i in range(len(bins)):
            cumulative_events[i] = np.sum(self.ct.data.mag > bins[i] - bin_size / 2)

        events_per_bin = np.abs(np.diff(np.append(cumulative_events, 0)))

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