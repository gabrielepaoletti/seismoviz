import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.axes import Axes
from matplotlib.collections import PathCollection


class BasePlotter:
    def __init__(self) -> None:
        pass

    def set_style(self, style: dict):
        """
        Sets a custom style for plotting.

        Parameters
        ----------
        style : dict
            A dictionary defining style attributes for the plot (e.g., 
            font size, line width, colors).
        """
        plt.rcParams.update(style)
    
    def reset_style(self):
        """
        Resets plotting style to the default one.
        """
        plt.rcdefaults()

    def plot_with_colorbar(
        self,
        ax: Axes,
        data: pd.DataFrame,
        x: str,
        y: str,
        color_by: str,
        cmap: str,
        edgecolor: str,
        size: float,
        alpha: float,
        cbar_orientation: str = 'horizontal',
        cbar_pad: float = 0.06,
        cbar_aspect: int = 40,
        cbar_shrink: float = 0.6,
    ) -> PathCollection:
        """
        Plots a scatter plot on the given axes with a colorbar.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object on which to draw the scatter plot.

        data : pd.DataFrame
            The data frame containing the data to be plotted.

        x : str
            The column name for the x-coordinates (e.g., longitude).

        y : str
            The column name for the y-coordinates (e.g., latitude).

        color_by : str
            The column name used for coloring the points.

        cmap : str
            The colormap to be used for the scatter plot.

        edgecolor : str
            Color of the point edges.

        size : float
            Size of the points.

        alpha : float
            Transparency of the points.

        Raises
        ------
        ValueError
            If the specified `color_by` column is not found in the data.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object created by `matplotlib`.
        """
        if color_by not in data.columns:
            raise ValueError(f"Column '{color_by}' not found in data.")

        colorbar_label = color_by
        if color_by == 'mag':
            colorbar_label = 'Magnitude'
        elif color_by == 'time':
            colorbar_label = 'Origin time'
        elif color_by == 'depth':
            colorbar_label = 'Depth [km]'

        color = data[color_by]
        if color_by == 'time':
            color_numeric = mdates.date2num(color)
            global_min = mdates.date2num(color.min())
            global_max = mdates.date2num(color.max())
        else:
            color_numeric = color
            global_min = np.floor(color.min())
            global_max = np.ceil(color.max())

        scatter = ax.scatter(
            x=data[x], y=data[y], c=color_numeric, cmap=cmap,
            edgecolor=edgecolor, s=size, alpha=alpha, linewidth=0.25,
            vmin=global_min, vmax=global_max, label=colorbar_label
        )

        cbar = plt.colorbar(
            scatter, ax=ax, orientation=cbar_orientation,
            pad=cbar_pad, shrink=cbar_shrink, aspect=cbar_aspect
        )
        cbar.set_label(colorbar_label, fontsize=14)

        if color_by == 'time':
            cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(cbar.ax.get_xticklabels(), rotation=45, ha='right')

        return scatter
    
    def save_figure(
        self, 
        save_name: str, 
        save_extension: str = 'jpg', 
        directory: str = './seismoviz_figures'
    ) -> None:
        """
        Saves the given figure to a file with the specified name, extension, 
        and directory.

        Parameters
        ----------
        save_name : str
            The base name used for saving the figure. It serves as the prefix 
            for the file name.

        save_extension : str, optional
            The file extension to use when saving figures (e.g., 'jpg', 'png'). 
            The default extension is 'jpg'.

        directory : str, optional
            The directory where the figure will be saved. Defaults to 
            './seismoviz_figures'.

        Returns
        -------
        None
        """
        os.makedirs(directory, exist_ok=True)
        fig_name = os.path.join(directory, f'{save_name}.{save_extension}')
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        print(f"Figure saved as {fig_name}")
