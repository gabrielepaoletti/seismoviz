import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cartopy.io.img_tiles import GoogleTiles
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from numpy.typing import ArrayLike
from matplotlib.text import Annotation, Text
from matplotlib.collections import PathCollection


class MapPlotter:
    def __init__(
        self, 
        projection=ccrs.Mercator(), 
        transform=ccrs.PlateCarree()
    ) -> None:
        """
        Initializes MapPlotter with default projection and transformation 
        settings.

        Parameters
        ----------
        projection : cartopy.crs projection, optional
            The map projection used to display the map. Defaults to 
            `ccrs.Mercator()`.

        transform : cartopy.crs projection, optional
            The coordinate reference system of the data to be plotted. 
            Defaults to `ccrs.PlateCarree()`.
        """
        plt.rc('axes', labelsize=14, titlesize=16)
        self.projection = projection
        self.transform = transform
        self.fig, self.ax = None, None

    def create_base_map(
            self,
            terrain_style: str = 'satellite',
            terrain_cmap: str = 'gray_r',
            terrain_alpha: str = 0.35,
            bounds_res: str = '50m',
            bmap_res: int = 12
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Creates a base map with common geographical features like land, 
        ocean, coastlines, and a satellite image background.

        Parameters
        ----------
        terrain_style : str, optional
            The style of the terrain background for the map. Common values 
            include 'satellite', 'terrain' or 'street'. Defaults to 'satellite'.

        terrain_cmap : str, optional
            The colormap to be applied to the terrain layer. Defaults to 'gray_r'.

        terrain_alpha : float, optional
            The transparency level for the terrain layer, where 0 is fully 
            transparent and 1 is fully opaque. Defaults to 0.35.

        bounds_res : str, optional
            The resolution for the geographical boundaries (such as 
            coastlines and borders) on the map. Common values are '10m', 
            '50m', or '110m'. Defaults to '50m'.

        bmap_res : int, optional
            The zoom level or resolution for the underlying map image 
            (e.g., satellite or terrain map). Defaults to 12.

        Returns
        -------
        matplotlib.figure.Figure
            The matplotlib figure object containing the map.

        matplotlib.axes._axes.Axes
            The matplotlib axes object configured with a cartographic 
            projection and geographical features.
        """
        fig = plt.figure(figsize=(10, 8), dpi=100)
        ax = plt.axes(projection=self.projection)

        terrain = GoogleTiles(desired_tile_form='L', style=terrain_style)
        ax.add_image(terrain, bmap_res, alpha=terrain_alpha)
        plt.set_cmap(terrain_cmap)

        ax.add_feature(cf.LAND.with_scale(bounds_res), color='white')
        ax.add_feature(cf.OCEAN.with_scale(bounds_res), color='lightblue')
        ax.add_feature(cf.COASTLINE.with_scale(bounds_res), lw=0.5)
        ax.add_feature(cf.BORDERS.with_scale(bounds_res), lw=0.3)

        gl = ax.gridlines(draw_labels=True, linewidth=1, alpha=0.5, linestyle=':')
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = {'size': 10}
        gl.ylabel_style = {'size': 10}

        gl.xformatter = LongitudeFormatter(degree_symbol='')
        gl.yformatter = LatitudeFormatter(degree_symbol='')

        return fig, ax

    def plot_with_colorbar(
            self,
            data: pd.DataFrame,
            x: str,
            y: str,
            color_by: str,
            cmap: str,
            edgecolor: str,
            size: float,
            alpha: float,
            legend: str
    ) -> PathCollection:
        """
        Plots data on the map with a colorbar.

        Parameters
        ----------
        data : pd.DataFrame
            The data frame containing the data to be plotted.

        x : str
            The column name for the x-coordinates (e.g., longitude).

        y : str
            The column name for the y-coordinates (e.g., latitude).

        color_by : str
            The column name used for coloring the points.

        cmap : str
            The color map to be used for the scatter plot.

        edgecolor : str
            Color of the point edges.

        size : float
            Size of the points.

        alpha : float
            Transparency of the points.

        legend : str
            Label for the scatter plot data, used for legend creation.

        Raises
        ------
        ValueError
            If the specified `color_by` column is not found in the data.

        Returns
        -------
        matplotlib.collections.PathCollection
            The scatter plot object created by `matplotlib`.
        """
        if self.fig is None or self.ax is None:
            self.fig, self.ax = self.create_base_map()

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

        scatter = self.scatter(
            x=data[x], y=data[y], c=color_numeric, cmap=cmap,
            edgecolor=edgecolor, s=size, alpha=alpha, linewidth=0.25,
            label=legend, vmin=global_min, vmax=global_max
        )

        cbar = plt.colorbar(
            scatter, ax=self.ax, orientation='horizontal',
            pad=0.06, shrink=0.6, aspect=40
        )
        cbar.set_label(colorbar_label, fontsize=14)

        if color_by == 'time':
            cbar.ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(cbar.ax.get_xticklabels(), rotation=45, ha='right')

        return scatter

    def extent(
            self,
            data: pd.DataFrame,
            xlim: tuple[float, float] = None, 
            ylim: tuple[float, float] = None
    ) -> tuple[float, float, float, float]:
        """
        Sets the map extent based on the provided data frame or explicit 
        limits.

        Parameters
        ----------
        data : pd.DataFrame
            A pandas DataFrame containing 'lon' and 'lat' columns for 
            longitude and latitude, respectively.

        xlim : tuple[float, float], optional
            A tuple specifying the minimum and maximum longitude to set 
            the map extent horizontally. Defaults to None.

        ylim : tuple[float, float], optional
            A tuple specifying the minimum and maximum latitude to set 
            the map extent vertically. Defaults to None.

        Returns
        -------
        tuple[float, float, float, float]
            The determined longitude and latitude bounds as 
            (lon_min, lon_max, lat_min, lat_max).
        """
        if xlim is None:
            lon_min, lon_max = data['lon'].min(), data['lon'].max()
        else:
            lon_min, lon_max = xlim

        if ylim is None:
            lat_min, lat_max = data['lat'].min(), data['lat'].max()
        else:
            lat_min, lat_max = ylim

        self.ax.set_extent(
            [lon_min - 0.001, lon_max + 0.001, lat_min - 0.001, lat_max + 0.001], 
            crs=self.transform
        )
        return lon_min, lon_max, lat_min, lat_max

    def inset(
            self,
            extent: tuple[float, float, float, float],
            buffer: int = 3, 
            inset_size: tuple[float, float] = (1.8, 1.8),
            bounds_res: str = '50m'
    ) -> plt.Axes:
        """
        Adds an inset map to the main map, showing a broader area around 
        the specified extent.

        Parameters
        ----------
        extent : tuple[float, float, float, float]
            A tuple specifying the extent of the main map.

        buffer : int, optional
            A buffer around the extent to determine the area shown in 
            the inset map. Default is 3 degrees.

        inset_size : tuple[float, float], optional
            The size of the inset map in figure coordinates. Default is 
            (1.8, 1.8).

        bounds_res : str, optional
            The resolution for the geographical boundaries on the map. 
            Defaults to '50m'.

        Returns
        -------
        matplotlib.axes.Axes
            The axes object of the inset map.
        """
        main_ax_bbox = self.ax.get_position()
        inset_left = main_ax_bbox.x1 - inset_size[0] / self.fig.get_figwidth() * (2.0 / 3.0)
        inset_bottom = main_ax_bbox.y1 - inset_size[1] / self.fig.get_figheight() * (2.0 / 3.0)
        inset_position = [
            inset_left, inset_bottom, 
            inset_size[0] / self.fig.get_figwidth(), 
            inset_size[1] / self.fig.get_figheight()
        ]

        inset_ax = self.fig.add_axes(inset_position, projection=self.projection)
        inset_extent = [
            extent[0] - buffer, extent[1] + buffer, 
            extent[2] - buffer, extent[3] + buffer
        ]
        inset_ax.set_extent(inset_extent, crs=self.transform)

        inset_ax.add_feature(cf.OCEAN.with_scale(bounds_res), color='lightblue')
        inset_ax.add_feature(cf.COASTLINE.with_scale(bounds_res), lw=0.5)
        inset_ax.add_feature(cf.BORDERS.with_scale(bounds_res), lw=0.3)

        inset_ax.plot(
            [extent[0], extent[1], extent[1], extent[0], extent[0]],
            [extent[2], extent[2], extent[3], extent[3], extent[2]],
            color='red', linewidth=1, transform=self.transform
        )

        return inset_ax

    def plot(
            self,
            x: ArrayLike,
            y: ArrayLike,
        **kwargs) -> list[plt.Line2D]:
        """
        Plots data on the map using matplotlib's plot method, applying 
        the specified transform.

        Parameters
        ----------
        x : ArrayLike
            The x-coordinates of the data to be plotted (e.g., longitude).

        y : ArrayLike
            The y-coordinates of the data to be plotted (e.g., latitude).

        **kwargs
            Additional keyword arguments passed to matplotlib's plot method.

        Returns
        -------
        list[plt.Line2D]
            A list of matplotlib.lines.Line2D objects representing the 
            plotted data.
        """
        plot = self.ax.plot(x, y, transform=self.transform, **kwargs)
        return plot

    def scatter(self, **kwargs) -> PathCollection:
        """
        Plots data points on the map using matplotlib's scatter method, 
        applying the specified transform.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to matplotlib's scatter 
            method.

        Returns
        -------
        matplotlib.collections.PathCollection
            A PathCollection object representing the plotted data points.
        """
        scatter = self.ax.scatter(transform=self.transform, **kwargs)
        return scatter

    def text(self, **kwargs) -> Text:
        """
        Adds text annotations to the map at specified locations, applying 
        the specified transform.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to matplotlib's text method.

        Returns
        -------
        matplotlib.text.Text
            A Text object representing the added annotation.
        """
        text = self.ax.text(transform=self.transform, **kwargs)
        return text

    def annotate(self, **kwargs) -> Annotation:
        """
        Adds annotations with optional arrows to the map, applying the 
        specified transform.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to matplotlib's annotate 
            method.

        Returns
        -------
        matplotlib.text.Annotation
            An Annotation object representing the added annotation.
        """
        annotate = self.ax.annotate(transform=self.transform, **kwargs)
        return annotate
