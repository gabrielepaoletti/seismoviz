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
        legend: str,
        cbar_orientation: str = 'horizontal',
        cbar_pad: float = 0.06,
        cbar_aspect: int = 40,
        cbar_shrink: float = 0.6,
    ) -> PathCollection:
        """
        Plots a scatter plot on the given axes with an associated colorbar.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object on which to draw the scatter plot.

        data : pd.DataFrame
            The data frame containing the data to be plotted. It should contain 
            the columns specified by `x`, `y`, and `color_by`.

        x : str
            The column name representing the x-coordinates (e.g., longitude).

        y : str
            The column name representing the y-coordinates (e.g., latitude).

        color_by : str
            The column name used to color the points in the scatter plot. 
            Values in this column will be mapped to the colormap specified 
            by `cmap`.

        cmap : str
            The colormap to use for mapping the values in the `color_by` 
            column to colors in the scatter plot.

        edgecolor : str
            The color of the edges of the points in the scatter plot.

        size : float
            The size of the points in the scatter plot.

        alpha : float
            The transparency level for the points in the scatter plot, 
            ranging from 0 (fully transparent) to 1 (fully opaque).

        legend : str
            Label for the scatter plot data, used for legend creation.

        cbar_orientation : str, optional
            Orientation of the colorbar, either 'horizontal' or 'vertical'. 
            Default is 'horizontal'.

        cbar_pad : float, optional
            Padding between the colorbar and the main plot, as a fraction 
            of the colorbar's height or width, depending on the orientation. 
            Default is 0.06.

        cbar_aspect : int, optional
            Aspect ratio of the colorbar (length to width ratio). 
            Default is 40.

        cbar_shrink : float, optional
            Factor by which to shrink the colorbar, as a fraction of the 
            original size. Default is 0.6.

        Raises
        ------
        ValueError
            If the specified `color_by` column is not found in the DataFrame.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object created by `matplotlib`, which can be 
            further modified or used for additional plotting.

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
            vmin=global_min, vmax=global_max, label=legend
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
